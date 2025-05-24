import argparse
import os
import sys
import time
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup


def is_valid_url(url, base_domain):
    parsed = urlparse(url)
    return parsed.netloc.endswith(base_domain) and parsed.scheme in {"http", "https"}


def get_links(soup, base_url, base_domain):
    links = set()
    for a in soup.find_all("a", href=True):
        href = urljoin(base_url, a["href"])
        cleaned = href.split("#")[0]
        if is_valid_url(cleaned, base_domain):
            links.add(cleaned)
    return links


def get_page_content(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        title = (
            soup.title.string.strip() if soup.title else urlparse(url).path.strip("/")
        )
        if not title:
            title = urlparse(url).netloc

        for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
            tag.decompose()

        text = " ".join(chunk.strip() for chunk in soup.stripped_strings)
        return title, text, soup
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None, None, None
    except Exception as e:
        print(f"Error processing {url}: {e}")
        return None, None, None


def get_category_from_url(url_string):
    parsed_url = urlparse(url_string)
    path_segments = parsed_url.path.strip("/").split("/")
    if path_segments and path_segments[0]:
        return path_segments[0]
    return "General"


def main(seed_url, max_pages_to_scrape, dest_dir):
    parsed_base = urlparse(seed_url)
    base_domain = parsed_base.netloc

    # --- Pre-scraping check for existing file ---
    # Define a representative filename for checking, even if final name differs slightly.
    # This uses max_pages_to_scrape to form a name for the check.
    potential_base_file_name = f"{base_domain.replace('.', '_').replace('-', '_')}_pages_{max_pages_to_scrape}.csv"
    potential_output_file_path = os.path.join(dest_dir, potential_base_file_name)

    os.makedirs(dest_dir, exist_ok=True)  # Ensure dest_dir exists before check

    if os.path.exists(potential_output_file_path):
        overwrite = (
            input(
                f"A file like {potential_output_file_path} (based on max_pages) already exists. \nIf the scrape results in {max_pages_to_scrape} pages, it will be overwritten. Continue? (y/N): "
            )
            .strip()
            .lower()
        )
        if overwrite != "y":
            print("Scraping aborted by user.")
            return
    # --- End pre-scraping check ---

    visited = set()
    queue = [seed_url]
    data = []

    print(
        f"Starting scrape of {seed_url} (domain: {base_domain}), up to {max_pages_to_scrape} pages."
    )

    while queue and len(data) < max_pages_to_scrape:
        url = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)
        print(f"Scraping: {url} ({len(data) + 1}/{max_pages_to_scrape})", end="\r")
        title, content, soup = get_page_content(url)

        if title and content:
            category = get_category_from_url(url)
            data.append(
                {
                    "id": url,
                    "title": title,
                    "url": url,
                    "content": content,
                    "category": category,
                }
            )

        if soup:
            new_links = get_links(soup, url, base_domain)
            for link in new_links:
                if link not in visited and link not in queue:
                    queue.append(link)

        time.sleep(0.5)

    print("\nScraping complete.")

    if not data:
        print("No data was scraped.")
        return

    df = pd.DataFrame(data)

    # Construct final output file path using actual number of pages scraped
    final_base_file_name = (
        f"{base_domain.replace('.', '_').replace('-', '_')}_pages_{len(data)}.csv"
    )
    final_output_file_path = os.path.join(dest_dir, final_base_file_name)

    # Note: The overwrite confirmation happened earlier based on potential_output_file_path.
    # If potential_output_file_path is different from final_output_file_path
    # (because len(data) != max_pages_to_scrape), the earlier check might not apply to the *exact* final file.
    # This is a compromise for simplicity. A more complex check could list all matching patterns.

    df.to_csv(final_output_file_path, index=False)
    print(f"\n✅ Saved {len(data)} pages to {final_output_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scrapes a website starting from a seed URL, up to a max number of pages, and saves content to a CSV."
    )
    parser.add_argument(
        "seed_url",
        type=str,
        help="The initial URL to start scraping from (e.g., https://example.com)",
    )
    parser.add_argument(
        "--max_pages",
        type=int,
        default=100,
        help="Maximum number of pages to scrape (default: 100)",
    )
    parser.add_argument(
        "--dest_dir",
        type=str,
        default="data/scraped",
        help="Destination directory for the output CSV file (default: data/scraped)",
    )

    args = parser.parse_args()

    if not args.seed_url.startswith(("http://", "https://")):
        print("Error: seed_url must start with http:// or https://")
        sys.exit(1)

    main(args.seed_url, args.max_pages, args.dest_dir)
