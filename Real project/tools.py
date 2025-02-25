import os
from typing import List, Literal
from langchain.tools import tool
#from langchain_community.tools.tavily_search import TavilySearchResults
from tavily import TavilyClient
from dotenv import load_dotenv
from logger import logger


load_dotenv()


tavily_tool = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def demo_mode(search_type: Literal["sas", "espc"]):
    if search_type == "sas":
        return "[{'topic': 'occupational therapy', 'results': [{'url': 'https://www.biography.com/athletes/lionel-messi', 'title': 'Lionel Messi: Biography, Soccer Player, Inter Miami CF, Athlete', 'content': 'Lionel Messi, a forward for Inter Miami CF, is one of the world’s greatest soccer players and helped the Argentina national team win its third FIFA World Cup in 2022. Messi, now playing for Inter Miami CF of the MLS, helped his home country win soccer’s biggest event for the first time since 1986, scoring two goals in the final and leading Argentina to a 4-2 win over Kylian Mbappé and France on penalties. Lionel Messi is an Argentinian soccer player who has played for FC Barcelona, Paris Saint-Germain, and currently, the MLS club Inter Miami CF as well as the Argentina national team.'}]}]"
    else:
        return """Lionel Messi, a forward for Inter Miami CF, is one of the world’s greatest soccer players and helped the Argentina national team win its third FIFA World Cup in 2022. Messi, now playing for Inter Miami CF of the MLS, helped his home country win soccer’s biggest event for the first time since 1986, scoring two goals in the final and leading Argentina to a 4-2 win over Kylian Mbappé and France on penalties. Lionel Messi is an Argentinian soccer player who has played for FC Barcelona, Paris Saint-Germain, and currently, the MLS club Inter Miami CF as well as the Argentina national team."""


#@tool
def search_and_summarize(query: str, 
                         topic: Literal["occupational therapy", "other"] = "occupational therapy"                         
                         ) -> List[str]:
    search_results_context = []
    
    search_response = tavily_tool.search(query, max_results=1, days=365, search_depth="basic")    
    search_results_context.append({
        "topic": topic,
        "results": [
            { "url": result["url"], "title": result["title"], "content": result["content"] } for result in search_response["results"]
        ]
    })    
    return search_results_context

def extract_search_page_content(url: str)-> str:    
    extract_response = tavily_tool.extract(url, include_images=False)
    return extract_response


if __name__ == "__main__":
    demoMode = True
    print ('Ready Player 1')

    if demoMode:
        logger.debug("Demo mode enabled")
        logger.debug(demo_mode("sas"))
        logger.debug(demo_mode("espc"))
    else:
        test_query = "Who is Leo Messi?"
        test_results = search_and_summarize(test_query)
        # Test the search_and_summarize function
        # If LLM says it would be helpful to summarize the content of the search results
        extract_search_page_content(test_results[0]["results"][0]["url"])
    # we will need to have the LLM make sure we want any of these top topics
        logger.debug(test_results)