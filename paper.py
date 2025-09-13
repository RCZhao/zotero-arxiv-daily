from typing import Optional
from functools import cached_property
from tempfile import TemporaryDirectory
import arxiv
import tarfile
import re
import time
from llm import get_llm
import requests
from requests.adapters import HTTPAdapter, Retry
from loguru import logger
import tiktoken
from contextlib import ExitStack
from urllib.error import HTTPError, ContentTooShortError



class ArxivPaper:
    def __init__(self,paper:arxiv.Result):
        self._paper = paper
        self.score = None
    
    @property
    def title(self) -> str:
        return self._paper.title
    
    @property
    def summary(self) -> str:
        return self._paper.summary
    
    @property
    def authors(self) -> list[str]:
        return self._paper.authors
    
    @cached_property
    def arxiv_id(self) -> str:
        return re.sub(r'v\d+$', '', self._paper.get_short_id())
    
    @property
    def pdf_url(self) -> str:
        return self._paper.pdf_url
    
    @cached_property
    def code_url(self) -> Optional[str]:
        s = requests.Session()
        retries = Retry(total=5, backoff_factor=0.1)
        s.mount('https://', HTTPAdapter(max_retries=retries))
        try:
            paper_list = s.get(f'https://paperswithcode.com/api/v1/papers/?arxiv_id={self.arxiv_id}').json()
        except Exception as e:
            logger.debug(f'Error when searching {self.arxiv_id}: {e}')
            return None

        if paper_list.get('count',0) == 0:
            return None
        paper_id = paper_list['results'][0]['id']

        try:
            repo_list = s.get(f'https://paperswithcode.com/api/v1/papers/{paper_id}/repositories/').json()
        except Exception as e:
            logger.debug(f'Error when searching {self.arxiv_id}: {e}')
            return None
        if repo_list.get('count',0) == 0:
            return None
        return repo_list['results'][0]['url']
    
    @cached_property
    def tex(self) -> dict[str,str]:
        with ExitStack() as stack:
            tmpdirname = stack.enter_context(TemporaryDirectory())
            
            max_retries = 3
            file = None
            for attempt in range(max_retries):
                try:
                    file = self._paper.download_source(dirpath=tmpdirname)
                    break  # Success
                except HTTPError as e:
                    if e.code == 404:
                        logger.warning(f"Source for {self.arxiv_id} not found (404). Skipping source analysis.")
                        return None
                    
                    wait_time = 5 * (attempt + 1)
                    logger.warning(f"Attempt {attempt + 1}/{max_retries} for {self.arxiv_id} failed with HTTPError {e.code}. Retrying in {wait_time}s...")
                    if attempt < max_retries - 1:
                        time.sleep(wait_time)
                except ContentTooShortError as e:
                    wait_time = 5 * (attempt + 1)
                    logger.warning(f"Attempt {attempt + 1}/{max_retries} for {self.arxiv_id} failed with incomplete download. Retrying in {wait_time}s...")
                    if attempt < max_retries - 1:
                        time.sleep(wait_time)

            if file is None:
                logger.error(f"Failed to download source for {self.arxiv_id} after {max_retries} attempts.")
                return None
            try:
                tar = stack.enter_context(tarfile.open(file))
            except tarfile.ReadError:
                logger.debug(f"Failed to find main tex file of {self.arxiv_id}: Not a tar file.")
                return None
 
            tex_files = [f for f in tar.getnames() if f.endswith('.tex')]
            if len(tex_files) == 0:
                logger.debug(f"Failed to find main tex file of {self.arxiv_id}: No tex file.")
                return None
            
            bbl_file = [f for f in tar.getnames() if f.endswith('.bbl')]
            match len(bbl_file) :
                case 0:
                    if len(tex_files) > 1:
                        logger.debug(f"Cannot find main tex file of {self.arxiv_id} from bbl: There are multiple tex files while no bbl file.")
                        main_tex = None
                    else:
                        main_tex = tex_files[0]
                case 1:
                    main_name = bbl_file[0].replace('.bbl','')
                    main_tex = f"{main_name}.tex"
                    if main_tex not in tex_files:
                        logger.debug(f"Cannot find main tex file of {self.arxiv_id} from bbl: The bbl file does not match any tex file.")
                        main_tex = None
                case _:
                    logger.debug(f"Cannot find main tex file of {self.arxiv_id} from bbl: There are multiple bbl files.")
                    main_tex = None
            if main_tex is None:
                logger.debug(f"Trying to choose tex file containing the document block as main tex file of {self.arxiv_id}")
            #read all tex files
            file_contents = {}
            for t in tex_files:
                f = tar.extractfile(t)
                content = f.read().decode('utf-8',errors='ignore')
                #remove comments
                content = re.sub(r'%.*\n', '\n', content)
                content = re.sub(r'\\begin{comment}.*?\\end{comment}', '', content, flags=re.DOTALL)
                content = re.sub(r'\\iffalse.*?\\fi', '', content, flags=re.DOTALL)
                #remove redundant \n
                content = re.sub(r'\n+', '\n', content)
                content = re.sub(r'\\\\', '', content)
                #remove consecutive spaces
                content = re.sub(r'[ \t\r\f]{3,}', ' ', content)
                if main_tex is None and re.search(r'\\begin\{document\}', content):
                    main_tex = t
                    logger.debug(f"Choose {t} as main tex file of {self.arxiv_id}")
                file_contents[t] = content
            
            if main_tex is not None:
                main_source:str = file_contents[main_tex]
                #find and replace all included sub-files
                include_files = re.findall(r'\\input\{(.+?)\}', main_source) + re.findall(r'\\include\{(.+?)\}', main_source)
                for f in include_files:
                    if not f.endswith('.tex'):
                        file_name = f + '.tex'
                    else:
                        file_name = f
                    main_source = main_source.replace(f'\\input{{{f}}}', file_contents.get(file_name, ''))
                file_contents["all"] = main_source
            else:
                logger.debug(f"Failed to find main tex file of {self.arxiv_id}: No tex file containing the document block.")
                file_contents["all"] = None
        return file_contents
    
    @cached_property
    def tldr(self) -> str:
        llm = get_llm()
        
        # Primary method: Use TeX source if available
        if self.tex is not None:
            introduction = ""
            conclusion = ""
            content = self.tex.get("all")
            if content is None:
                content = "\n".join(self.tex.values())
            
            # Clean and extract
            content = re.sub(r'~?\\cite.?\{.*?\}', '', content)
            content = re.sub(r'\\begin\{figure\}.*?\\end\{figure\}', '', content, flags=re.DOTALL)
            content = re.sub(r'\\begin\{table\}.*?\\end\{table\}', '', content, flags=re.DOTALL)
            intro_match = re.search(r'\\section\{Introduction\}.*?(\\section|\\end\{document\}|\\bibliography|\\appendix|$)', content, flags=re.DOTALL)
            if intro_match:
                introduction = intro_match.group(0)
            concl_match = re.search(r'\\section\{Conclusion\}.*?(\\section|\\end\{document\}|\\bibliography|\\appendix|$)', content, flags=re.DOTALL)
            if concl_match:
                conclusion = concl_match.group(0)

            # Only proceed if we have introduction
            if introduction:
                logger.debug(f"Generating TLDR for {self.arxiv_id} using TeX source.")
                prompt = """Given the title, abstract, introduction and the conclusion (if any) of a paper in latex format, generate a one-sentence TLDR summary in __LANG__:
        
                \\title{__TITLE__}
                \\begin{abstract}__ABSTRACT__\\end{abstract}
                __INTRODUCTION__
                __CONCLUSION__
                """
                prompt = prompt.replace('__LANG__', llm.lang)
                prompt = prompt.replace('__TITLE__', self.title)
                prompt = prompt.replace('__ABSTRACT__', self.summary)
                prompt = prompt.replace('__INTRODUCTION__', introduction)
                prompt = prompt.replace('__CONCLUSION__', conclusion)

                try:
                    enc = tiktoken.encoding_for_model("gpt-4o")
                    prompt_tokens = enc.encode(prompt)
                    prompt_tokens = prompt_tokens[:4000]
                    prompt = enc.decode(prompt_tokens)

                    tldr = llm.generate(
                        messages=[
                            {
                                "role": "system",
                                "content": "You are an assistant who perfectly summarizes scientific paper, and gives the core idea of the paper to the user.",
                            },
                            {"role": "user", "content": prompt},
                        ]
                    )
                    if tldr and tldr.strip():
                        return tldr # Success with primary method
                except Exception as e:
                    logger.warning(f"Failed to generate TLDR for {self.arxiv_id} from TeX source: {e}. Falling back to abstract-based generation.")

        # Fallback method: Use title and abstract
        logger.debug(f"Generating TLDR for {self.arxiv_id} using title and abstract only.")
        prompt = """Given the title and abstract of a paper, generate a one-sentence TLDR summary in __LANG__:
    
        Title: __TITLE__
        Abstract: __ABSTRACT__
        """
        prompt = prompt.replace('__LANG__', llm.lang)
        prompt = prompt.replace('__TITLE__', self.title)
        prompt = prompt.replace('__ABSTRACT__', self.summary)

        try:
            enc = tiktoken.encoding_for_model("gpt-4o")
            prompt_tokens = enc.encode(prompt)
            prompt_tokens = prompt_tokens[:4000]
            prompt = enc.decode(prompt_tokens)

            tldr = llm.generate(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an assistant who perfectly summarizes scientific paper, and gives the core idea of the paper to the user.",
                    },
                    {"role": "user", "content": prompt},
                ]
            )
            
            if tldr and tldr.strip():
                note = " (Generated from title and abstract only)"
                if llm.lang.lower() == 'chinese':
                    note = " (仅根据标题和摘要生成)"
                return tldr.strip() + note
            else:
                raise ValueError("LLM returned an empty or whitespace-only TLDR.")

        except Exception as e:
            # Final fallback: return the abstract itself
            logger.error(f"Failed to generate TLDR for {self.arxiv_id} from abstract: {e}. Returning abstract as TLDR.")
            note = " (AI summary failed, showing abstract)"
            if llm.lang.lower() == 'chinese':
                note = " (AI摘要生成失败，显示原文摘要)"
            return self.summary + note

    @cached_property
    def affiliations(self) -> Optional[list[str]]:
        # First, try to extract from TeX source for higher accuracy
        if self.tex is not None:
            content = self.tex.get("all")
            if content is None:
                content = "\n".join(self.tex.values())
            
            # Search for author information in the TeX content
            possible_regions = [r'\\author.*?\\maketitle', r'\\begin{document}.*?\\begin{abstract}']
            matches = [re.search(p, content, flags=re.DOTALL) for p in possible_regions]
            match = next((m for m in matches if m), None)
            
            if match:
                information_region = match.group(0)
                prompt = f"Given the author information of a paper in latex format, extract the affiliations of the authors in a python list format, which is sorted by the author order. If there is no affiliation found, return an empty list '[]'. Following is the author information:\n{information_region}"
                
                enc = tiktoken.encoding_for_model("gpt-4o")
                prompt_tokens = enc.encode(prompt)
                prompt_tokens = prompt_tokens[:4000]
                prompt = enc.decode(prompt_tokens)
                
                llm = get_llm()
                affiliations_str = llm.generate(
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an assistant who perfectly extracts affiliations of authors from the author information of a paper. You should return a python list of affiliations sorted by the author order, like ['TsingHua University','Peking University']. If an affiliation is consisted of multi-level affiliations, like 'Department of Computer Science, TsingHua University', you should return the top-level affiliation 'TsingHua University' only. Do not contain duplicated affiliations. If there is no affiliation found, you should return an empty list [ ]. You should only return the final list of affiliations, and do not return any intermediate results.",
                        },
                        {"role": "user", "content": prompt},
                    ]
                )

                try:
                    # Parse the LLM output
                    affiliations_str = re.search(r'\[.*?\]', affiliations_str, flags=re.DOTALL).group(0)
                    affiliations_list = eval(affiliations_str)
                    affiliations_list = list(set(affiliations_list))
                    affiliations_list = [str(a) for a in affiliations_list]
                    if affiliations_list:
                        logger.debug(f"Extracted affiliations for {self.arxiv_id} from TeX source.")
                        return affiliations_list
                except Exception as e:
                    logger.debug(f"Failed to parse LLM output for affiliations of {self.arxiv_id} from TeX: {e}")
            else:
                logger.debug(f"Failed to find author region in TeX for {self.arxiv_id}.")

        # Fallback to arXiv API data if TeX parsing fails or yields no results
        logger.debug(f"Falling back to arXiv API for affiliations of {self.arxiv_id}.")
        try:
            # The arxiv library stores the raw feedparser entry in _raw
            author_details = self._paper._raw.get('authors', [])
            api_affiliations = []
            for author in author_details:
                if 'arxiv_affiliation' in author:
                    # The affiliation text is in the 'term' key
                    api_affiliations.append(author['arxiv_affiliation']['term'])
            
            if api_affiliations:
                # Deduplicate and return
                unique_affiliations = sorted(list(set(api_affiliations)))
                logger.debug(f"Extracted affiliations for {self.arxiv_id} from API.")
                return unique_affiliations
            else:
                logger.debug(f"No affiliation data found in arXiv API for {self.arxiv_id}.")
                return None
        except Exception as e:
            logger.error(f"Error extracting affiliations from arXiv API for {self.arxiv_id}: {e}")
            return None
