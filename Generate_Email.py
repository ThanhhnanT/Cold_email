from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import  OutputParserException
from Secret_key import GROQ_COLD_EMAIL

class Chain():
    def __init__(self):
        self.llm = ChatGroq(
            temperature=0,
            groq_api_key = GROQ_COLD_EMAIL,
            model_name = 'llama-3.3-70b-versatile'
        )

    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {data}

            ### INSTRUCTION:
            The scraped text is from the career's page of a website.
            Your job is to extract job postings and return a **list** of JSON objects.
            Each object must have these fields: 'role', 'experience', 'skills', and 'description', 'companyName'.

            ### FORMAT:
            [
              {{
                "companyName" : "...",
                "role": "...",
                "experience": "...",
                "skills": [...],
                "description": "..."
              }},
              ...
            ]

            ### JSON ONLY (NO EXPLANATION)
            """
        )

        chain_extract = prompt_extract | self.llm
        response = chain_extract.invoke(input={'data': cleaned_text})
        print("Raw LLM response:", response.content)  # Debug
        try:
            json_parser = JsonOutputParser()
            result = json_parser.parse(response.content)
        except OutputParserException:
            raise OutputParserException('Context too big or output format invalid')

        return result if isinstance(result, list) else [result]
    def write_email(self, job, links):
        prompt_email = PromptTemplate.from_template(
            """
            ### MÔ TẢ CÔNG VIỆC:
            {job_description}

            ### HƯỚNG DẪN:
            Bạn là VVT, sinh viên năm cuối trường Đại học Bách khoa Hà Nội (HUST) với GPA 3.35. 
            Nhiệm vụ của bạn là viết một email xin việc gửi tới nhà tuyển dụng dựa trên mô tả công việc ở trên.
            Trong email, hãy thể hiện sự quan tâm đến vị trí này, nêu bật kỹ năng, kinh nghiệm và nền tảng học tập phù hợp với yêu cầu công việc.
            Bạn cũng có thể đưa vào các đường link phù hợp nhất từ danh sách portfolio sau để làm nổi bật năng lực cá nhân: {link_list}.
            Email cần ngắn gọn, lịch sự, thể hiện thiện chí và không có phần giải thích ngoài nội dung email.

            ### EMAIL (KHÔNG CẦN MỞ ĐẦU GIẢI THÍCH):
            """
        )

        chain_email = prompt_email | self.llm
        res = chain_email.invoke({"job_description": str(job), "link_list": links})
        return res.content