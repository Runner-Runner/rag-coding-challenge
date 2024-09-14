import os

from langchain_community.vectorstores import FAISS


base_queries = [
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

extended_simple_queries = [
    *base_queries,
    ("Wie viel wiegt XBO 4000 W/HS XL OFR?",
     'ZMP_1007199_XBO_4000_W_HS_XL_OFR.pdf',
     '1022,90 g'),
    ("Wie viel wiegt XBO 1600 W/HSC XL OFR?",
     'ZMP_1007177_XBO_1600_W_HSC_XL_OFR.pdf',
     '321,00 g'),
    ("Wie viel wiegt XBO 3000 W/H XL OFR?",
     'ZMP_1007191_XBO_3000_W_H_XL_OFR.pdf',
     '687,00 g'),
    ("Wie viel wiegt XBO 4500 W/HS XL OFR?",
     'ZMP_1007201_XBO_4500_W_HS_XL_OFR.pdf',
     '1023,00 g'),
    ("Wie viel wiegt XBO 10000 W/HS OFR?",
     'ZMP_55851_XBO_10000_W_HS_OFR.pdf',
     '1030,00 g'),

    ("Welche Leuchte hat SCIP Nummer dd2ddf15-037b-4473-8156-97498e721fb3?",
     'ZMP_1007193_XBO_3000_W_HS_XL_OFR.pdf',
     'dd2ddf15-037b-4473-8156-97498e721fb3'),
    ("Welche Leuchte hat SCIP Nummer 7934C4E7-AD5D-4E20-9BAA-2349E600F6AD?",
     'ZMP_55851_XBO_10000_W_HS_OFR.pdf',
     '7934C4E7-AD5D-4E20-9BAA-2349E600F6AD'),
    ("Welche Leuchte hat SCIP Nummer 0b6df52f-0b97-4d4d-bcc0-6d1dab9f13b9?",
     'ZMP_55877_XBO_4000_W_HSA_OFR.pdf',
     '0b6df52f-0b97-4d4d-bcc0-6d1dab9f13b9'),
    ("Welche Leuchte hat SCIP Nummer e8ef51c8-36d5-458d-aa43-bff7de28bede?",
     'ZMP_1200637_XBO_2000_W_HS_OFR.pdf',
     'e8ef51c8-36d5-458d-aa43-bff7de28bede'),
    ("Welche Leuchte hat SCIP Nummer 1ed097d6-f451-4c5c-b29a-3f156fd7504f?",
     'ZMP_1007209_XBO_7000_W_HS_XL_OFR.pdf',
     '1ed097d6-f451-4c5c-b29a-3f156fd7504f'),

    ("Welche Leuchte hat die Erzeugnissnummer 4008321299963?",
     'ZMP_1007189_XBO_2500_W_HS_XL_OFR.pdf',
     '4008321299963'),
    ("Welche Leuchte hat die Erzeugnissnummer 4008321412928?",
     'ZMP_1007209_XBO_7000_W_HS_XL_OFR.pdf',
     '4008321412928'),
    ("Welche Leuchte hat die Erzeugnissnummer 4008321412980?",
     'ZMP_55877_XBO_4000_W_HSA_OFR.pdf',
     '4008321412980'),
    ("Welche Leuchte hat die Erzeugnissnummer 4008321650436?",
     'ZMP_1007195_XBO_3000_W_HTP_XL_OFR.pdf',
     '4008321650436'),
    ("Welche Leuchte hat die Erzeugnissnummer 4008321412829?",
     'ZMP_1007203_XBO_5000_W_H_XL_OFR.pdf',
     '4008321412829'),

    # Other query options
    ("Was hat XBO 4000 W/HS XL OFR für eine Nennspannung?",
     'ZMP_1007199_XBO_4000_W_HS_XL_OFR.pdf',
     '23,0 V'),
    ("Welchen Durchmesser hat XBO 4000 W/HS XL OFR?",
     'ZMP_1007199_XBO_4000_W_HS_XL_OFR.pdf',
     '46,0 mm'),
    ("Welche Lebensdauer hat XBO 4000 W/HS XL OFR?",
     'ZMP_1007199_XBO_4000_W_HS_XL_OFR.pdf',
     '2500 h'),
    ("Wie lang ist XBO 4000 W/HS XL OFR mit Sockel?",
     'ZMP_1007199_XBO_4000_W_HS_XL_OFR.pdf',
     '222,00 mm'),
    ("Welches Volument hat XBO 4000 W/HS XL OFR?",
     'ZMP_1007199_XBO_4000_W_HS_XL_OFR.pdf',
     '13.58 dm³'),
]


def test_base_simple_queries(vector_store: FAISS):
    output_test_query_similarities(vector_store, base_queries)


def test_extended_simple_queries(vector_store: FAISS):
    output_test_query_similarities(vector_store, extended_simple_queries)


def output_test_query_similarities(vector_store: FAISS, queries):
    total_count = len(queries)
    best_hit_count = 0
    top4_hit_count = 0
    content_hit_count = 0
    for query, true_relevant_doc, expected_content in queries:
        results = vector_store.similarity_search(query, k=4)
        top4_docs = [os.path.basename(r.metadata['source']) for r in results]
        best_hit = true_relevant_doc == top4_docs[0]
        best_hit_count += best_hit
        top4_hit = true_relevant_doc in top4_docs
        top4_hit_count += top4_hit

        # Does the given context chunk of the document contain the expected information?
        content_hit = False
        page_content = ''
        if top4_hit:
            # TODO Could be improved: only checks "most relevant" chunk for correct doc and ignores other chunks from
            #  the same doc
            page_content = results[top4_docs.index(true_relevant_doc)].page_content
            content_hit = expected_content.lower() in page_content.lower()
            page_content = '(' + page_content + ')'
        content_hit_count += content_hit

        print("{}: [{}] Korrektes Doc, [{}] Korrekt in Top 4, [{}] Korrekter page_content {}".format(
            query, 'x' if best_hit else ' ', 'x' if top4_hit else ' ', 'x' if content_hit else ' ', page_content))
    print(f"Gesamt: "
          f"\nKorrektes Doc: {(best_hit_count / total_count):.2f} ({best_hit_count}/{total_count})"
          f"\nKorrekt in Top 4: {(top4_hit_count / total_count):.2f} ({top4_hit_count}/{total_count})"
          f"\nKorrekter page_content: {(content_hit_count / total_count):.2f} ({content_hit_count}/{total_count})")
