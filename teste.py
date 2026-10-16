import chromadb

client = chromadb.PersistentClient("C:/chroma_data/order_reviews")
col = client.get_collection("orders_reviews")
print("Total:", col.count())
print(col.peek(1)["documents"])
res = col.query(query_texts=None, query_embeddings=[col.peek(1)["embeddings"][0]], n_results=3)
print("Consulta OK:", len(res["ids"][0]))