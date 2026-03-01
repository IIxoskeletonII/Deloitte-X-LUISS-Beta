"""
demo.py
-------
Role: Interactive Demo Interface. Provides a Gradio web application with
two main tabs for exploring the semantic search system:

  Tab 1 - Semantic Search: Enter a natural-language query with optional
    filters (minimum rating, category, number of results) and receive
    ranked product results with similarity scores, summaries, and entities.

  Tab 2 - Product Explorer: Select a product to view its full details,
    review summary, entity tags, and discover similar product recommendations.

Launches with share=True to generate a public URL suitable for Google Colab.

Usage: Run from root via module mode: python -m src.demo
"""

import json
import gradio as gr
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.search_engine import SearchEngine


def format_entities(entities):
    if not entities:
        return "No entities extracted"
    if isinstance(entities, str):
        try:
            entities = json.loads(entities)
        except (json.JSONDecodeError, TypeError):
            return entities
    return ", ".join(f"{k}: {v}" for k, v in entities.items())


def create_demo():
    print("[*] Initializing Search Engine for Demo...")
    engine = SearchEngine()
    categories = engine.get_categories()
    category_choices = ["All"] + categories

    product_titles = engine.metadata[["parent_asin", "product_title"]].copy()
    product_titles["display"] = product_titles["parent_asin"] + " | " + product_titles["product_title"].astype(str).str[:80]
    product_choices = product_titles["display"].tolist()[:5000]

    def do_search(query, min_rating, category, num_results):
        if not query or not query.strip():
            return pd.DataFrame()

        min_r = float(min_rating) if min_rating and min_rating > 0 else None
        cat = category if category and category != "All" else None

        results = engine.search(query.strip(), top_k=int(num_results), min_rating=min_r, category=cat)

        if not results:
            return pd.DataFrame({"Message": ["No results found. Try adjusting your filters."]})

        rows = []
        for r in results:
            rows.append({
                "Score": f"{r['score']:.3f}",
                "Product": r["product_title"],
                "Rating": r["average_rating"],
                "Store": r["store"],
                "Summary": str(r["summary"])[:200] if r["summary"] else "",
                "Entities": format_entities(r["entities"])[:150],
            })
        return pd.DataFrame(rows)

    def get_details(product_selection):
        if not product_selection:
            return "Select a product to view details.", pd.DataFrame()

        asin = product_selection.split(" | ")[0].strip()
        details = engine.get_product_details(asin)

        if details is None:
            return "Product not found.", pd.DataFrame()

        info_parts = [
            f"Product: {details.get('product_title', 'N/A')}",
            f"Store: {details.get('store', 'N/A')}",
            f"Average Rating: {details.get('average_rating', 'N/A')}",
            f"Number of Ratings: {details.get('rating_number', 'N/A')}",
            f"Category: {details.get('main_category', 'N/A')}",
            "",
            f"--- Review Summary ---",
            f"{details.get('summary', 'No summary available.')}",
            "",
            f"--- Entities ---",
            f"{format_entities(details.get('entities', {}))}",
        ]
        info_text = "\n".join(info_parts)

        recs = engine.recommend(asin, top_k=5)
        if recs:
            rec_rows = []
            for r in recs:
                rec_rows.append({
                    "Score": f"{r['score']:.3f}",
                    "Product": r["product_title"],
                    "Rating": r["average_rating"],
                    "Store": r["store"],
                })
            rec_df = pd.DataFrame(rec_rows)
        else:
            rec_df = pd.DataFrame({"Message": ["No recommendations available."]})

        return info_text, rec_df

    with gr.Blocks(title="DiscoverAI - Semantic Product Search") as app:
        gr.Markdown("# DiscoverAI: Review-Aware Semantic Search")
        gr.Markdown("Search Health & Personal Care products by meaning and sentiment, powered by NLP.")

        with gr.Tab("Semantic Search"):
            with gr.Row():
                with gr.Column(scale=3):
                    query_input = gr.Textbox(
                        label="Search Query",
                        placeholder="e.g., low-priced skincare product for sensitive skin",
                        lines=1,
                    )
                with gr.Column(scale=1):
                    num_results = gr.Slider(
                        minimum=1, maximum=25, value=10, step=1,
                        label="Number of Results",
                    )
            with gr.Row():
                min_rating_slider = gr.Slider(
                    minimum=0, maximum=5, value=0, step=0.5,
                    label="Minimum Average Rating (0 = no filter)",
                )
                category_dropdown = gr.Dropdown(
                    choices=category_choices,
                    value="All",
                    label="Category Filter",
                )
            search_btn = gr.Button("Search", variant="primary")
            results_table = gr.DataFrame(label="Search Results")

            search_btn.click(
                fn=do_search,
                inputs=[query_input, min_rating_slider, category_dropdown, num_results],
                outputs=[results_table],
            )
            query_input.submit(
                fn=do_search,
                inputs=[query_input, min_rating_slider, category_dropdown, num_results],
                outputs=[results_table],
            )

        with gr.Tab("Product Explorer"):
            product_dropdown = gr.Dropdown(
                choices=product_choices,
                label="Select a Product",
                filterable=True,
            )
            explore_btn = gr.Button("View Details", variant="primary")
            product_info = gr.Textbox(label="Product Details", lines=12, interactive=False)
            rec_table = gr.DataFrame(label="Similar Products (Recommendations)")

            explore_btn.click(
                fn=get_details,
                inputs=[product_dropdown],
                outputs=[product_info, rec_table],
            )

    return app


def main():
    print("[*] Launching DiscoverAI Demo...")
    app = create_demo()
    app.launch(share=True)


if __name__ == "__main__":
    main()
