import pandas as pd
import chromadb
import uuid

class Project:
    def __init__(self, root = "./my_portfolio.csv"):
        self.root = root
        self.data = pd.read_csv(root)
        self.chroma_client = chromadb.PersistentClient('vectorstore')
        self.collection = self.chroma_client.get_or_create_collection(name='project')

    def load_project(self):
        if not self.collection.count():
            for _,row in self.data.iterrows():
                self.collection.add(
                    documents=row['Techstack'],
                    metadatas= {"links": row['Links']},
                    ids=[str(uuid.uuid4())]
                )

    def query(self, skills):
        links = self.collection.query(
            query_texts=skills,
            n_results=4
        ).get('metadatas', [])
        return links