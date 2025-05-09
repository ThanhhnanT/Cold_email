import streamlit as st
from Secret_key import CTAI
from Project import Project
from Clean_Text import clean_text
from langchain_community.document_loaders import WebBaseLoader
from Generate_Email import Chain

def Create_streamlit(llm, project, clear):
    print(1)
    st.title("Cold Mail Generator")

    url_input = st.text_input("Enter a URL:", value=CTAI)
    submit_button = st.button("Submit")

    if submit_button:
        try:
            loader = WebBaseLoader(url_input)
            data = loader.load()[0].page_content
            project.load_project()
            jobs = llm.extract_jobs(data)
            for job in jobs:
                skills = job.get('skills', [])
                print(skills)
                links = project.query(skills)
                email = llm.write_email(job, links)
                st.code(email, language='markdown')
        except Exception as e:
            st.error(f"An Error Occurred: {e}")


if __name__ =='__main__':
    chain = Chain()
    project = Project()
    st.set_page_config(layout='wide', page_title='Cold Email Generator')
    Create_streamlit(chain, project, clean_text)