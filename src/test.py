import os

from langchain_community.vectorstores import FAISS


def test_base_query_similarities(vector_store: FAISS):
    # English version
    labeled_queries = [
        ("Wie viel wiegt XBO 4000 W/HS XL OFR?",
         'ZMP_1007199_XBO_4000_W_HS_XL_OFR.pdf',
         '1022,90 g'),
        ("Welche Leuchte hat SCIP Nummer dd2ddf15-037b-4473-8156-97498e721fb3?",
         'ZMP_1007193_XBO_3000_W_HS_XL_OFR.pdf',
         'dd2ddf15-037b-4473-8156-97498e721fb3'),
        ("Welche Leuchte hat die Erzeugnissnummer 4008321299963?",
         'ZMP_1007189_XBO_2500_W_HS_XL_OFR.pdf',
         '4008321299963'),
    ]
    total_count = len(labeled_queries)
    best_hit_count = 0
    top4_hit_count = 0
    content_hit_count = 0
    for query, true_relevant_doc, expected_content in labeled_queries:
        results = vector_store.similarity_search(query, k=4)
        top4_docs = [os.path.basename(r.metadata['source']) for r in results]
        best_hit = true_relevant_doc == top4_docs[0]
        best_hit_count += best_hit
        top4_hit = true_relevant_doc in top4_docs
        top4_hit_count += top4_hit

        # Does the given context chunk of the document contain the expected information?
        content_hit = False
        if top4_hit:
            expected_content in results[top4_docs.index(true_relevant_doc)].page_content
            content_hit = True
        content_hit_count += content_hit

        print("{}: [{}] Korrektes Doc, [{}] Korrekt in Top 4, [{}] Korrekter page_content".format(
            query, 'x' if best_hit else ' ', 'x' if top4_hit else ' ', 'x' if content_hit else ' '))
    print(f"Gesamt: "
          f"\nKorrektes Doc: {(best_hit_count / total_count):.2f} ({best_hit_count}/{total_count})"
          f"\nKorrekt in Top 4: {(top4_hit_count / total_count):.2f} ({top4_hit_count}/{total_count})"
          f"\nKorrekter page_content: {(content_hit_count / total_count):.2f} ({content_hit_count}/{total_count})")
