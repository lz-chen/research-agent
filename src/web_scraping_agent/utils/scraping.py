import os
import requests
import firecrawl
import dotenv
from typing import List, Callable, Dict
from prettytable import PrettyTable, ALL
from tqdm import tqdm
from web_scraping_agent.utils.tokens import calculate_cost
from web_scraping_agent.utils.config import settings

# dotenv.load_dotenv()
#
#
# class CrawlerSettings(BaseSettings):
#     # use pydantic settings to get the api key
#     FIRECRAWL_API_KEY: str
#
#
# settings = CrawlerSettings()


def scrape_jina_ai(url: str) -> str:
    response = requests.get("https://r.jina.ai/" + url)
    return response.text


def scrape_firecrawl(url: str):
    app = firecrawl.FirecrawlApp(api_key=settings.FIRECRAWL_API_KEY)
    scraped_data = app.scrape_url(url)["markdown"]
    return scraped_data


def view_scraped_content(scrape_url_functions: List[Dict[str, Callable[[str], str]]], sites_list: List[Dict[str, str]],
                         characters_to_display: int = 500, table_max_width: int = 50) -> List[Dict[str, str]]:
    content_table_headers = ["Site Name"] + [f"{func['name']} content" for func in scrape_url_functions]
    cost_table_headers = ["Site Name"] + [f"{func['name']} cost" for func in scrape_url_functions]

    content_table = PrettyTable()
    content_table.field_names = content_table_headers

    cost_table = PrettyTable()
    cost_table.field_names = cost_table_headers

    scraped_data = []

    for site in sites_list:
        content_row = [site['name']]
        cost_row = [site['name']]
        site_data = {"provider": site['name'], "sites": []}

        for scrape_function in scrape_url_functions:
            function_name = scrape_function['name']
            for _ in tqdm([site], desc=f"Processing site {site['name']} using {function_name}"):
                try:
                    content = scrape_function['function'](site['url'])
                    content_snippet = content[:characters_to_display]
                    content_row.append(content_snippet)

                    cost = calculate_cost(content)
                    cost_row.append(f"${cost:.6f}")

                    site_data["sites"].append({"name": function_name, "content": content})
                except Exception as e:
                    error_message = f"Error: {str(e)}"
                    content_row.append(error_message)
                    cost_row.append("Error")

                    site_data["sites"].append({"name": function_name, "content": error_message})
                    continue

        content_table.add_row(content_row)
        cost_table.add_row(cost_row)
        scraped_data.append(site_data)

    content_table.max_width = table_max_width
    content_table.hrules = ALL

    cost_table.max_width = table_max_width
    cost_table.hrules = ALL

    print("Content Table:")
    print(content_table)

    print("\nCost Table:\nThis is how much it would cost to use gpt-4o to parse this content for extraction.")
    print(cost_table)

    return scraped_data


def extract(client, user_input: str, system_message: str):
    entity_extraction_system_message = {"role": "system", "content": system_message}
    messages = [entity_extraction_system_message]
    messages.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        stream=False,
        response_format={"type": "json_object"}
    )

    return response.choices[0].message.content
