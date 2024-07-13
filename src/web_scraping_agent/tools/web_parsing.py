from web_scraping_agent.utils.scraping import scrape_firecrawl, scrape_jina_ai
from llama_index.core.tools import FunctionTool
from crewai_tools import LlamaIndexTool
from web_scraping_agent.utils.config import settings

# def parse_member_page(client, page_url: str, sys_msg: str, scraper_func: callable = scrape_firecrawl, ):
#     """
#     This function scrapes a page containing information about members of a company.
#     :param page_url: str, the url to scrape
#     :return:
#     """
#     member_page_content = scraper_func(page_url)
#     # print(member_page_content)
#     extracted_member_info = extract(client, member_page_content, sys_msg)
#     return extracted_member_info
#
#
# def parse_company_page(client, page_url: str, sys_msg: str, scraper_func: callable = scrape_firecrawl, ):
#     """
#     This function scrapes the main page of a company.
#     :param page_url: str, the url to scrape
#     :return:
#     """
#     company_page_content = scraper_func(page_url)
#     extracted_company_info = extract(client, company_page_content, sys_msg)
#     return extracted_company_info


def check_member_page_url_validity(members_page_url: str, company_page_url: str):
    """
    This function checks if the url of the members page if valid.
    Usually the members page is in the same domain as the company page.
    :param members_page_url: str, the url of the members page
    :param company_page_url: str, the url of the company page
    :return:
    """
    members_page_url = members_page_url.strip("https://").strip("http://")
    company_page_url = company_page_url.strip("https://").strip("http://")
    if company_page_url in members_page_url:
        return True
    return False


def crawl_web_page(page_url: str):
    """
    This function scrapes the content of a given url.
    :param page_url: str, the url to scrape
    :return:
    """
    if settings.CRAWLER == "firecrawl":
        return scrape_firecrawl(page_url)
    elif settings.CRAWLER == "jina":
        return scrape_jina_ai(page_url)
    else:
        raise NotImplementedError(f"Crawler {settings.CRAWLER} not implemented")


li_scraper_tool = FunctionTool.from_defaults(fn=crawl_web_page)
li_url_validation_tool = FunctionTool.from_defaults(fn=check_member_page_url_validity)

cai_scraper_tool = LlamaIndexTool.from_tool(li_scraper_tool)
cai_url_validation_tool = LlamaIndexTool.from_tool(li_url_validation_tool)

# li_tools = {
#     "parse_member_page": FunctionTool.from_defaults(fn=parse_member_page),
#     "parse_company_page": FunctionTool.from_defaults(fn=parse_company_page),
#     "check_member_page_url_validity": FunctionTool.from_defaults(fn=check_member_page_url_validity)
# }
#
# cai_tools = {k: LlamaIndexTool.from_tool(v) for k, v in li_tools.items()}
