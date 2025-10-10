import logging
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from sqlalchemy import create_engine, text, MetaData, Table, inspect, and_, or_
from sqlalchemy.exc import SQLAlchemyError
import json
from decimal import Decimal
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from pathlib import Path
from datetime import datetime, date
import google.generativeai as genai
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import asyncio
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import io
import base64
from typing import Dict, List, Any
import uuid

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s')
log = logging.getLogger("university_rag")

app = FastAPI(title="Professional University Database Management System")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Database configuration
DATABASE_URL = "postgresql+psycopg2://postgres:0608@localhost:5432/university_db"
engine = create_engine(DATABASE_URL, echo=False)
templates = Jinja2Templates(directory="templates")

# RAG Configuration
RAG_DATA_PATH = "nl2sql.csv"
FAISS_INDEX_PATH = "faiss_index"
EMBEDDINGS_PATH = "embeddings.pkl"
RAG_DATA_PKL_PATH = "rag_data.pkl"
SIMILARITY_THRESHOLD = 0.85

# Gemini API Configuration
GEMINI_API_KEY = "AIzaSyAoxOnUJ9PMaWhLVouG5pfentMhUfVlI5Y"

# T5 Model Configuration
T5_MODEL_NAME = 'cssupport/t5-small-awesome-text-to-sql'

# Database Schema
SCHEMA = """
CREATE TABLE department (
  department_id INT PRIMARY KEY,
  department_name VARCHAR(100) UNIQUE,
  established_date DATE,
  hod_name VARCHAR(100),
  contact_email VARCHAR(100)
);
CREATE TABLE student (
  student_id INT PRIMARY KEY,
  first_name VARCHAR(50),
  last_name VARCHAR(50),
  gender VARCHAR(10),
  dob DATE,
  email VARCHAR(100) UNIQUE,
  phone VARCHAR(20),
  address VARCHAR(200),
  admission_year INT,
  department_id INT REFERENCES department(department_id),
  created_date TIMESTAMP
);
CREATE TABLE faculty (
  faculty_id INT PRIMARY KEY,
  first_name VARCHAR(50),
  last_name VARCHAR(50),
  email VARCHAR(100) UNIQUE,
  phone VARCHAR(20),
  designation VARCHAR(50),
  qualification VARCHAR(100),
  joining_date DATE,
  department_id INT REFERENCES department(department_id),
  salary FLOAT,
  is_active BOOLEAN
);
CREATE TABLE course (
  course_id INT PRIMARY KEY,
  course_name VARCHAR(100),
  course_code VARCHAR(20) UNIQUE,
  credits INT,
  duration_years INT,
  description VARCHAR(500),
  department_id INT REFERENCES department(department_id)
);
CREATE TABLE subject (
  subject_id INT PRIMARY KEY,
  subject_name VARCHAR(100),
  subject_code VARCHAR(20) UNIQUE,
  credits INT,
  semester INT,
  course_id INT REFERENCES course(course_id),
  faculty_id INT REFERENCES faculty(faculty_id)
);
CREATE TABLE enrollment (
  enrollment_id INT PRIMARY KEY,
  student_id INT REFERENCES student(student_id),
  course_id INT REFERENCES course(course_id),
  enroll_date DATE,
  academic_year INT,
  semester INT,
  is_active BOOLEAN
);
CREATE TABLE marks (
  marks_id INT PRIMARY KEY,
  student_id INT REFERENCES student(student_id),
  subject_id INT REFERENCES subject(subject_id),
  internal_marks FLOAT,
  external_marks FLOAT,
  total_marks FLOAT,
  grade VARCHAR(2),
  exam_date DATE,
  semester INT
);
CREATE TABLE attendance (
  attendance_id INT PRIMARY KEY,
  student_id INT REFERENCES student(student_id),
  subject_id INT REFERENCES subject(subject_id),
  total_classes INT,
  attended_classes INT,
  attendance_date DATE,
  semester INT,
  attendance_percentage FLOAT
);
CREATE TABLE admin_user (
  user_id INT PRIMARY KEY,
  username VARCHAR(50) UNIQUE,
  password VARCHAR(100),
  role VARCHAR(20),
  department_id INT REFERENCES department(department_id),
  full_name VARCHAR(100),
  email VARCHAR(100),
  is_active BOOLEAN,
  created_date TIMESTAMP
);
""".strip()

class EnhancedRAGSystem:
    def __init__(self):
        self.model = None
        self.index = None
        self.rag_data = None
        self.is_initialized = False
        self.t5_tokenizer = None
        self.t5_model = None
        self.gemini_model = None
        
    def initialize_rag(self):
        """Initialize the RAG system with FAISS index and AI models"""
        try:
            # Initialize T5 model for SQL generation
            log.info("Loading T5 model for SQL generation...")
            self.t5_tokenizer = T5Tokenizer.from_pretrained(T5_MODEL_NAME)
            self.t5_model = T5ForConditionalGeneration.from_pretrained(T5_MODEL_NAME)
            log.info("T5 model loaded successfully")
            
            # Initialize Gemini
            try:
                if GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_GEMINI_API_KEY":
                    genai.configure(api_key=GEMINI_API_KEY)
                    self.gemini_model = genai.GenerativeModel('gemini-2.5-flash')
                    log.info("Gemini model configured successfully")
                else:
                    log.warning("Gemini API key not configured. Gemini fallback will be disabled.")
                    self.gemini_model = None
            except Exception as e:
                log.warning(f"Failed to initialize Gemini: {e}")
                self.gemini_model = None
            
            # Check if FAISS index already exists
            if (Path(FAISS_INDEX_PATH).exists() and 
                Path(EMBEDDINGS_PATH).exists() and 
                Path(RAG_DATA_PKL_PATH).exists()):
                
                log.info("Loading existing FAISS index and RAG data...")
                self.index = faiss.read_index(FAISS_INDEX_PATH)
                
                with open(EMBEDDINGS_PATH, 'rb') as f:
                    self.embeddings = pickle.load(f)
                    
                with open(RAG_DATA_PKL_PATH, 'rb') as f:
                    self.rag_data = pickle.load(f)
                    
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                self.is_initialized = True
                log.info("RAG system initialized from existing files")
                return True
                
            # Create new FAISS index if it doesn't exist
            if not Path(RAG_DATA_PATH).exists():
                log.warning(f"RAG data file {RAG_DATA_PATH} not found. RAG system will be disabled.")
                return False
                
            log.info("Creating new FAISS index for RAG system...")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Load and preprocess RAG data
            df = pd.read_csv(RAG_DATA_PATH)
            self.rag_data = df.to_dict('records')
            
            # Generate embeddings
            natural_language_queries = [item['src_nl'] for item in self.rag_data]
            self.embeddings = self.model.encode(natural_language_queries, normalize_embeddings=True)
            
            # Create FAISS index
            dimension = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
            self.index.add(self.embeddings.astype('float32'))
            
            # Save the index and data
            faiss.write_index(self.index, FAISS_INDEX_PATH)
            with open(EMBEDDINGS_PATH, 'wb') as f:
                pickle.dump(self.embeddings, f)
            with open(RAG_DATA_PKL_PATH, 'wb') as f:
                pickle.dump(self.rag_data, f)
                
            self.is_initialized = True
            log.info(f"RAG system initialized with {len(self.rag_data)} entries")
            return True
            
        except Exception as e:
            log.error(f"Error initializing RAG system: {e}")
            return False
    
    def search_similar_query(self, query, threshold=SIMILARITY_THRESHOLD):
        """Search for similar queries in RAG database"""
        if not self.is_initialized:
            return None
            
        try:
            query_embedding = self.model.encode([query], normalize_embeddings=True).astype('float32')
            similarities, indices = self.index.search(query_embedding, k=1)
            
            if similarities[0][0] >= threshold:
                best_match_idx = indices[0][0]
                best_match = self.rag_data[best_match_idx]
                return {
                    'sql_query': best_match['tgt_sql'],
                    'similarity': float(similarities[0][0]),
                    'source_query': best_match['src_nl'],
                    'method': 'rag'
                }
            return None
            
        except Exception as e:
            log.error(f"Error searching in RAG system: {e}")
            return None
    
    def generate_sql_with_t5(self, question):
        """Generate SQL using T5 model"""
        try:
            input_prompt = f"tables:\n{SCHEMA}\nquery for: {question}"
            inputs = self.t5_tokenizer(input_prompt, padding=True, truncation=True, return_tensors="pt")
            outputs = self.t5_model.generate(
                **inputs, 
                max_length=512, 
                num_beams=4, 
                early_stopping=True
            )
            generated_sql = self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return {
                'sql_query': generated_sql,
                'method': 't5',
                'confidence': 0.7
            }
            
        except Exception as e:
            log.error(f"Error generating SQL with T5: {e}")
            return None
    
    def generate_sql_with_gemini(self, question):
        """Generate SQL using Gemini as fallback"""
        if not self.gemini_model:
            return None
            
        try:
            prompt = f"""
            Given the following database schema:
            
            {SCHEMA}
            
            Convert this natural language query to PostgreSQL SQL: "{question}"
            
            Return only the SQL query without any explanation or markdown formatting.
            Make sure the query is syntactically correct and uses proper PostgreSQL syntax.
            """
            
            response = self.gemini_model.generate_content(prompt)
            generated_sql = response.text.strip()
            
            if generated_sql.startswith('```sql'):
                generated_sql = generated_sql[6:]
            if generated_sql.startswith('```'):
                generated_sql = generated_sql[3:]
            if generated_sql.endswith('```'):
                generated_sql = generated_sql[:-3]
            generated_sql = generated_sql.strip()
            
            return {
                'sql_query': generated_sql,
                'method': 'gemini',
                'confidence': 0.6
            }
            
        except Exception as e:
            log.error(f"Error generating SQL with Gemini: {e}")
            return None

class DatabaseManager:
    def __init__(self, engine):
        self.engine = engine
        self.metadata = MetaData()
        self.metadata.reflect(bind=engine)
    
    def get_table_names(self):
        """Get all table names"""
        return self.metadata.tables.keys()
    
    def get_table_data(self, table_name, limit=1000, filters=None):
        """Get data from specific table with filters"""
        try:
            table = Table(table_name, self.metadata, autoload_with=self.engine)
            query = table.select().limit(limit)
            
            if filters:
                for column, value in filters.items():
                    if value and value != '':
                        if isinstance(value, dict):
                            # Handle range filters
                            if 'min' in value and value['min'] != '':
                                query = query.where(table.columns[column] >= float(value['min']))
                            if 'max' in value and value['max'] != '':
                                query = query.where(table.columns[column] <= float(value['max']))
                        else:
                            # Handle text filters
                            query = query.where(table.columns[column].ilike(f"%{value}%"))
            
            with self.engine.connect() as conn:
                result = conn.execute(query)
                columns = result.keys()
                rows = result.fetchall()
                
                data = []
                for row in rows:
                    row_dict = {}
                    for i, col in enumerate(columns):
                        value = row[i]
                        if isinstance(value, (datetime, date)):
                            value = value.isoformat()
                        elif isinstance(value, Decimal):
                            value = float(value)
                        row_dict[col] = value
                    data.append(row_dict)
                
                return {
                    "success": True,
                    "table_name": table_name,
                    "columns": list(columns),
                    "data": data,
                    "row_count": len(data)
                }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_table_schema(self, table_name):
        """Get schema information for a table"""
        try:
            inspector = inspect(self.engine)
            columns = inspector.get_columns(table_name)
            foreign_keys = inspector.get_foreign_keys(table_name)
            
            schema_info = {
                "table_name": table_name,
                "columns": [],
                "foreign_keys": foreign_keys
            }
            
            for column in columns:
                schema_info["columns"].append({
                    "name": column['name'],
                    "type": str(column['type']),
                    "nullable": column['nullable'],
                    "primary_key": column.get('primary_key', False)
                })
            
            return schema_info
        except Exception as e:
            return {"error": str(e)}
    
    def get_advanced_filters_data(self, table_name, filters):
        """Get filtered data with advanced filtering options"""
        try:
            table = Table(table_name, self.metadata, autoload_with=self.engine)
            query = table.select()
            
            conditions = []
            for column, filter_data in filters.items():
                if filter_data.get('value') and filter_data.get('value') != '':
                    if filter_data.get('operator') == 'equals':
                        conditions.append(table.columns[column] == filter_data['value'])
                    elif filter_data.get('operator') == 'contains':
                        conditions.append(table.columns[column].ilike(f"%{filter_data['value']}%"))
                    elif filter_data.get('operator') == 'starts_with':
                        conditions.append(table.columns[column].ilike(f"{filter_data['value']}%"))
                    elif filter_data.get('operator') == 'ends_with':
                        conditions.append(table.columns[column].ilike(f"%{filter_data['value']}"))
                    elif filter_data.get('operator') == 'greater_than':
                        conditions.append(table.columns[column] > float(filter_data['value']))
                    elif filter_data.get('operator') == 'less_than':
                        conditions.append(table.columns[column] < float(filter_data['value']))
                    elif filter_data.get('operator') == 'between':
                        if filter_data.get('min_value') and filter_data.get('max_value'):
                            conditions.append(table.columns[column].between(float(filter_data['min_value']), float(filter_data['max_value'])))
            
            if conditions:
                query = query.where(and_(*conditions))
            
            with self.engine.connect() as conn:
                result = conn.execute(query)
                columns = result.keys()
                rows = result.fetchall()
                
                data = []
                for row in rows:
                    row_dict = {}
                    for i, col in enumerate(columns):
                        value = row[i]
                        if isinstance(value, (datetime, date)):
                            value = value.isoformat()
                        elif isinstance(value, Decimal):
                            value = float(value)
                        row_dict[col] = value
                    data.append(row_dict)
                
                return {
                    "success": True,
                    "table_name": table_name,
                    "columns": list(columns),
                    "data": data,
                    "row_count": len(data)
                }
        except Exception as e:
            return {"success": False, "error": str(e)}

class VisualizationEngine:
    def __init__(self):
        self.color_palette = px.colors.qualitative.Set3
    
    def create_matplotlib_plot(self, data, x_col, y_col, plot_type='bar'):
        """Create matplotlib plot and return as base64 image"""
        try:
            df = pd.DataFrame(data)
            
            plt.figure(figsize=(10, 6))
            
            if plot_type == 'bar':
                plt.bar(df[x_col].astype(str), df[y_col].astype(float))
                plt.title(f'{y_col} by {x_col}')
                plt.xlabel(x_col)
                plt.ylabel(y_col)
                plt.xticks(rotation=45)
                
            elif plot_type == 'line':
                plt.plot(df[x_col].astype(str), df[y_col].astype(float), marker='o')
                plt.title(f'{y_col} by {x_col}')
                plt.xlabel(x_col)
                plt.ylabel(y_col)
                plt.xticks(rotation=45)
                
            elif plot_type == 'scatter':
                plt.scatter(df[x_col].astype(float), df[y_col].astype(float))
                plt.title(f'{y_col} vs {x_col}')
                plt.xlabel(x_col)
                plt.ylabel(y_col)
                
            elif plot_type == 'pie':
                plt.pie(df[y_col].astype(float), labels=df[x_col].astype(str), autopct='%1.1f%%')
                plt.title(f'{y_col} Distribution')
                
            elif plot_type == 'histogram':
                plt.hist(df[y_col].astype(float), bins=10, alpha=0.7, edgecolor='black')
                plt.title(f'Distribution of {y_col}')
                plt.xlabel(y_col)
                plt.ylabel('Frequency')
            
            plt.tight_layout()
            
            # Convert plot to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return f"data:image/png;base64,{image_base64}"
            
        except Exception as e:
            log.error(f"Error creating matplotlib plot: {e}")
            return None
    
    def create_plotly_visualization(self, data, x_col, y_col, plot_type='bar', title=None):
        """Create Plotly visualization"""
        try:
            df = pd.DataFrame(data)
            
            if plot_type == 'bar':
                fig = px.bar(df, x=x_col, y=y_col, title=title or f'{y_col} by {x_col}')
            elif plot_type == 'line':
                fig = px.line(df, x=x_col, y=y_col, title=title or f'{y_col} by {x_col}')
            elif plot_type == 'scatter':
                fig = px.scatter(df, x=x_col, y=y_col, title=title or f'{y_col} vs {x_col}')
            elif plot_type == 'pie':
                fig = px.pie(df, names=x_col, values=y_col, title=title or f'{y_col} Distribution')
            elif plot_type == 'histogram':
                fig = px.histogram(df, x=y_col, title=title or f'Distribution of {y_col}')
            else:
                return None
            
            return fig.to_json()
            
        except Exception as e:
            log.error(f"Error creating plotly visualization: {e}")
            return None
    
    def recommend_visualization(self, data, columns):
        """Recommend the best visualization type based on data"""
        df = pd.DataFrame(data)
        
        numeric_columns = []
        categorical_columns = []
        
        for col in columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_columns.append(col)
            else:
                categorical_columns.append(col)
        
        if len(numeric_columns) >= 2:
            return 'scatter', numeric_columns[0], numeric_columns[1]
        elif len(categorical_columns) >= 1 and len(numeric_columns) >= 1:
            return 'bar', categorical_columns[0], numeric_columns[0]
        elif len(numeric_columns) >= 1:
            return 'histogram', numeric_columns[0], numeric_columns[0]
        else:
            return 'bar', columns[0], columns[1] if len(columns) > 1 else columns[0]

# Initialize systems
rag_system = EnhancedRAGSystem()
db_manager = DatabaseManager(engine)
viz_engine = VisualizationEngine()

@app.on_event("startup")
async def startup_event():
    rag_system.initialize_rag()

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        return super().default(obj)

# Routes for different pages
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/nl2sql", response_class=HTMLResponse)
async def nl2sql_page(request: Request):
    return templates.TemplateResponse("nl2sql.html", {"request": request})

@app.get("/filter", response_class=HTMLResponse)
async def filter_page(request: Request):
    tables = db_manager.get_table_names()
    return templates.TemplateResponse("filter.html", {"request": request, "tables": list(tables)})

@app.get("/api/tables")
async def get_tables():
    tables = db_manager.get_table_names()
    return {"tables": list(tables)}

@app.get("/api/table-schema/{table_name}")
async def get_table_schema(table_name: str):
    schema = db_manager.get_table_schema(table_name)
    return schema

@app.get("/api/table-data/{table_name}")
async def get_table_data(table_name: str, limit: int = 100):
    result = db_manager.get_table_data(table_name, limit)
    return result

@app.post("/api/table-data")
async def get_table_data_post(request: Request):
    data = await request.json()
    table_name = data.get('table_name')
    filters = data.get('filters', {})
    limit = data.get('limit', 1000)
    
    result = db_manager.get_table_data(table_name, limit, filters)
    return result

@app.post("/api/advanced-filter")
async def advanced_filter(request: Request):
    data = await request.json()
    table_name = data.get('table_name')
    filters = data.get('filters', {})
    
    result = db_manager.get_advanced_filters_data(table_name, filters)
    return result

@app.post("/rag-query")
async def rag_query(request: Request):
    try:
        data = await request.json()
        natural_language_query = data.get('query', '').strip()
        
        if not natural_language_query:
            return {"success": False, "error": "Query cannot be empty"}
        
        sql_result = None
        method_used = "unknown"
        confidence = 0.0
        fallback_used = False
        
        # Step 1: Try RAG first
        rag_result = rag_system.search_similar_query(natural_language_query)
        
        if rag_result:
            sql_result = rag_result
            method_used = "rag"
            confidence = rag_result['similarity']
        else:
            # Step 2: Try T5 model
            t5_result = rag_system.generate_sql_with_t5(natural_language_query)
            if t5_result and t5_result['sql_query']:
                sql_result = t5_result
                method_used = "t5"
                confidence = t5_result['confidence']
                fallback_used = True
            else:
                # Step 3: Try Gemini as final fallback
                gemini_result = rag_system.generate_sql_with_gemini(natural_language_query)
                if gemini_result and gemini_result['sql_query']:
                    sql_result = gemini_result
                    method_used = "gemini"
                    confidence = gemini_result['confidence']
                    fallback_used = True
                else:
                    return {
                        "success": False, 
                        "error": "No suitable SQL generation method found.",
                        "natural_language_query": natural_language_query
                    }
        
        if not sql_result:
            return {
                "success": False, 
                "error": "Failed to generate SQL query",
                "natural_language_query": natural_language_query
            }
        
        sql_query = sql_result['sql_query']
        
        # Execute the SQL
        try:
            with engine.connect() as conn:
                result = conn.execute(text(sql_query))
                columns = result.keys()
                rows = result.fetchall()
                
                data_rows = []
                for row in rows:
                    row_dict = {}
                    for i, col in enumerate(columns):
                        value = row[i]
                        if isinstance(value, (datetime, date)):
                            value = value.isoformat()
                        elif isinstance(value, Decimal):
                            value = float(value)
                        elif value is None:
                            value = None
                        row_dict[col] = value
                    data_rows.append(row_dict)
                
                response_data = {
                    "success": True,
                    "columns": list(columns),
                    "data": data_rows,
                    "row_count": len(data_rows),
                    "generated_sql": sql_query,
                    "natural_language_query": natural_language_query,
                    "method_used": method_used,
                    "confidence": confidence,
                    "fallback_used": fallback_used
                }
                
                if method_used == 'rag':
                    response_data.update({
                        "rag_similarity": sql_result['similarity'],
                        "rag_source_query": sql_result['source_query']
                    })
                
                return response_data
                
        except SQLAlchemyError as e:
            return {
                "success": False,
                "error": f"Database error: {str(e)}",
                "generated_sql": sql_query,
                "natural_language_query": natural_language_query,
                "method_used": method_used,
                "fallback_used": fallback_used
            }
            
    except Exception as e:
        return {"success": False, "error": f"Error: {str(e)}"}

@app.post("/api/visualize")
async def create_visualization(request: Request):
    data = await request.json()
    viz_type = data.get('type')
    chart_data = data.get('data')
    options = data.get('options', {})
    
    try:
        if viz_type == 'matplotlib':
            result = viz_engine.create_matplotlib_plot(
                chart_data,
                options.get('x_col'),
                options.get('y_col'),
                options.get('plot_type', 'bar')
            )
            return {"success": True, "plot_image": result}
        else:
            result = viz_engine.create_plotly_visualization(
                chart_data,
                options.get('x_col'),
                options.get('y_col'),
                viz_type,
                options.get('title')
            )
            return {"success": True, "plot_json": result}
            
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/api/recommend-visualization")
async def recommend_visualization(request: Request):
    data = await request.json()
    chart_data = data.get('data')
    columns = data.get('columns')
    
    try:
        plot_type, x_col, y_col = viz_engine.recommend_visualization(chart_data, columns)
        return {
            "success": True,
            "recommendation": {
                "plot_type": plot_type,
                "x_col": x_col,
                "y_col": y_col
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)