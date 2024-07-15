PMT_CRAWL_MAIN_PAGE = """
You help user to get the main information about the company,
 including their goal, detailed summary of the company, established time, 
 the link to the page which contains information about the members of the company
  (usually a site within the same domain). 
 Return as a JSON with four keys: 
 {company_goal: str, company_summary: str, 
 established_time: str, members_page: str}."""

PMT_CRAWL_INFO_PAGE = """
You help user to get the company members mentioned in the given page content.
 Return as a JSON with the key 'members' containing a list of dictionaries
  with keys 'name', 'role' and 'linkedin_page'. 
  Note that when linkedIn page is not available, 
  it should be an empty string."""

PMT_CRAWL_ALL = """
You help user to get the main information about a company from provided website, including
 their goal, detailed summary of the company, established time, and the key members 
 (Founder, CEO, COO, CTO etc.) of the company. Return your result as a JSON with four keys:
 company_goal, company_summary, established_time, and key_members. key_members contains a list of
 dictionaries with keys 'name', 'role' and 'linkedin_page'. 
 If the linkedIn page is not directly available on the given page content,
  it should be an empty string, do not try to come up with made up linkedin profile.
"""