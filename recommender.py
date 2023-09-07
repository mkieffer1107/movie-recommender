import os
import vecs
import time
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from transformers import pipeline
model = pipeline("feature-extraction", model="Supabase/gte-small")

# define colors 
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
WHITE = "\033[97m"
END_COLOR = "\033[0m"

# load environment variables from .env file
load_dotenv()

# supabase db connection
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
db_pass: str = os.environ.get("DATABASE_PASSWORD")
db_connection = "postgresql://postgres:[YOUR-PASSWORD]@db.pedbaridbklowihouaqa.supabase.co:5432/postgres".replace("[YOUR-PASSWORD]", db_pass)

print(f"\n{YELLOW}Connecting to database{END_COLOR}\n")
vx = vecs.create_client(db_connection)
docs = vx.get_or_create_collection(name="movies", dimension=384)

# number of entries in the supabase table
NUM_MOVIES = 34886

# type of embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")


def query_db(queries, top_k):
    # get top k results
    top_k = min(top_k, NUM_MOVIES) 
    print(f"{YELLOW}Getting top {top_k} results for {len(queries)} queries{END_COLOR}\n")

    # index the collection for fast search performance
    docs.create_index()

    # make queries to database
    results = {}
    for query in queries:
        # get query embedding
        query_embedding = embedder.encode(query)

        # get results from supabase db
        result = docs.query(
            data=query_embedding,            # embedding to search
            limit=top_k,                     # number of records to return
            filters={},                      # metadata filters -- none right now
            include_metadata=True,           # include metadata in results -- {title, wiki page}
        )
        
        # store results in a dictionary
        results[query] = result

    # disconnect from the database
    vx.disconnect()
    return results



def print_results(results, elapsed):
    print(f"{MAGENTA}==================================================={END_COLOR}")

    for i, query in enumerate(queries):
        if i != 0:
            print(f"{MAGENTA}---------------------------------------------------{END_COLOR}")
        print(f"{YELLOW}Query:{END_COLOR} {query}\n")

        query_results = results[query]

        print(f"{CYAN}Movies:{END_COLOR}")
        for result in query_results:
            print(f" - {result[1]['title']}")

    print(f"{MAGENTA}==================================================={END_COLOR}")
    print(f"{GREEN}total runtime: {elapsed:.3f}s{END_COLOR}\n")

if __name__ == "__main__":
    top_k = 5
    queries = ["jack gets a beanstalk and a giant golden egg", 
               "sup dawg",
               "a guy shoots 100 guys",
               "child goes on magical adventure",
               "animated pirates fight over treasure",
               "7 magical balls",
               "bryan mills's daughter gets taken in france",
               "bryan mills's ex-wife gets killed in france",
               "tom hanks is santa claus",
               "pineapple express",] 
    
    print(f"{YELLOW}Queries:{END_COLOR}")
    for query in queries:
        print(f"{GREEN} - {query}{END_COLOR}")
    print()

    start = time.time()
    results = query_db(queries, top_k)
    end = time.time()

    print_results(results, end-start)