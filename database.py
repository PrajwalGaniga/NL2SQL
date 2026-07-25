from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Same connection string pattern from previous project, updated to the new DB name
DATABASE_URL = "postgresql+psycopg2://postgres:0608@localhost:5432/nosqlStudentDB"

# Create database engine
engine = create_engine(DATABASE_URL, echo=False)

def test_connection():
    """Verify that the database connection is working."""
    try:
        with engine.connect() as connection:
            logger.info("Successfully connected to the database: nosqlStudentDB")
            return True
    except SQLAlchemyError as e:
        logger.error(f"Failed to connect to the database. Error: {e}")
        return False

if __name__ == "__main__":
    test_connection()
