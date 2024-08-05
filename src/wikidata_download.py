import mwclient
import datetime
import json

# Connect to the Wikipedia site
site = mwclient.Site('en.wikipedia.org')

# Specify the cutoff date
cutoff_date = datetime.datetime(2022, 9, 1)

# Function to convert time.struct_time to datetime.datetime
def struct_time_to_datetime(struct_time):
    return datetime.datetime(
        struct_time.tm_year,
        struct_time.tm_mon,
        struct_time.tm_mday,
        struct_time.tm_hour,
        struct_time.tm_min,
        struct_time.tm_sec
    )

# Function to check if an article was edited after the cutoff date
def was_edited_after(page, cutoff_date):
    for rev in page.revisions(limit=1, dir='newer'):
        rev_date = struct_time_to_datetime(rev['timestamp'])
        if rev_date >= cutoff_date:
            return True
    return False

# Function to fetch recent articles
def fetch_recent_articles(cutoff_date, limit=100, max_articles=10):
    recent_articles = []
    counter = 0
    added_articles = 0
    print("Starting to fetch recent articles...")
    for page in site.recentchanges(namespace=0, toponly=True, limit=limit):
        counter += 1
        page_obj = site.pages[page['title']]
        if was_edited_after(page_obj, cutoff_date):
            text = page_obj.text()
            recent_articles.append({"title": page['title'], "text": text})
            added_articles += 1
            print(f"Added article: {page['title']}")
            if added_articles >= max_articles:
                print(f"Reached the limit of {max_articles} articles.")
                break
        else:
            print(f"Skipped article: {page['title']} (edited before cutoff date)")
        
        if counter % 10 == 0:
            print(f"Processed {counter} articles so far...")
    
    print(f"Total articles fetched: {len(recent_articles)}")
    return recent_articles

# Fetch articles edited after the cutoff date
recent_articles = fetch_recent_articles(cutoff_date, limit=100, max_articles=10)

# Save articles to a JSON file
with open('recent_articles.json', 'w', encoding='utf-8') as f:
    json.dump(recent_articles, f, ensure_ascii=False, indent=4)

print("Fetched and saved recent articles edited after", cutoff_date)