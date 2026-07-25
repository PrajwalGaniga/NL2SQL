from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
# pyrefly: ignore [missing-import]
import google.generativeai as genai
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
import os
from dotenv import load_dotenv
from database import engine
from datetime import datetime, date
from decimal import Decimal

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("WARNING: GEMINI_API_KEY not found in .env")

model = genai.GenerativeModel("gemini-2.5-flash")

app = FastAPI(title="NoSQL to SQL App")
templates = Jinja2Templates(directory="templates")

# The database schema for context
DB_SCHEMA = """
CREATE TABLE courses (
    course_id SERIAL PRIMARY KEY,
    course_name VARCHAR(100) NOT NULL,
    duration_months INTEGER NOT NULL,
    course_fee DECIMAL(10,2),
    instructor_name VARCHAR(100),
    is_active BOOLEAN DEFAULT TRUE
);
CREATE TABLE students (
    student_id SERIAL PRIMARY KEY,
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50),
    age INTEGER,
    gender VARCHAR(10),
    email VARCHAR(100) UNIQUE,
    phone VARCHAR(15),
    city VARCHAR(50),
    admission_date DATE,
    cgpa DECIMAL(3,2),
    course_id INTEGER,
    FOREIGN KEY (course_id) REFERENCES courses(course_id) ON DELETE CASCADE
);
"""

class QueryRequest(BaseModel):
    query: str

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/query")
async def process_query(request: QueryRequest):
    natural_language = request.query.strip()
    
    if not natural_language:
        return {"success": False, "error": "Query cannot be empty"}
        
    try:
        # Construct the prompt for Gemini
        prompt = f"""
        You are an expert PostgreSQL developer. 
        Given the following database schema:
        
        {DB_SCHEMA}
        
        Translate this natural language query to an exact PostgreSQL SQL query: "{natural_language}"
        
        Return ONLY the SQL query without any explanation, markdown formatting, or code blocks.
        Make sure the query is syntactically correct, handles joins if needed, and uses proper PostgreSQL syntax.
        """
        
        response = model.generate_content(prompt)
        sql_query = response.text.strip()
        
        # Strip potential markdown formatting if Gemini includes it despite instructions
        if sql_query.startswith("```sql"):
            sql_query = sql_query[6:]
        if sql_query.startswith("```"):
            sql_query = sql_query[3:]
        if sql_query.endswith("```"):
            sql_query = sql_query[:-3]
        sql_query = sql_query.strip()
        
        # Execute the SQL Query
        with engine.connect() as conn:
            result = conn.execute(text(sql_query))
            columns = list(result.keys())
            rows = result.fetchall()
            
            # Serialize the data (handle dates, decimals, etc.)
            data_rows = []
            for row in rows:
                row_dict = {}
                for col_idx, col_name in enumerate(columns):
                    val = row[col_idx]
                    if isinstance(val, (datetime, date)):
                        val = val.isoformat()
                    elif isinstance(val, Decimal):
                        val = float(val)
                    row_dict[col_name] = val
                data_rows.append(row_dict)
                
            return {
                "success": True,
                "sql_query": sql_query,
                "columns": columns,
                "data": data_rows,
                "row_count": len(data_rows)
            }
            
    except SQLAlchemyError as e:
        return {
            "success": False,
            "error": f"Database error during execution: {str(e)}",
            "sql_query": sql_query if 'sql_query' in locals() else None
        }
    except Exception as e:
        return {
            "success": False, 
            "error": f"An error occurred: {str(e)}"
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)