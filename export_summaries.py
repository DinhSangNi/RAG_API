"""
Script to export summary documents from database as markdown files
"""

import os
from datetime import datetime
from pathlib import Path
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from app.database.models import SummaryDocument
from app.config import settings

def export_summaries_to_markdown():
    """Export all summary documents from database to markdown files"""
    
    # Create output directory
    output_dir = Path("./exported_summaries")
    output_dir.mkdir(exist_ok=True)
    
    # Create database session
    engine = create_engine(settings.DATABASE_URL)
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()
    
    try:
        # Query all summary documents
        summaries = db.query(SummaryDocument).all()
        
        if not summaries:
            print("No summary documents found in database.")
            return
        
        print(f"Found {len(summaries)} summary documents. Exporting...")
        
        success_count = 0
        for idx, summary in enumerate(summaries, 1):
            try:
                # Create filename from ID or index
                filename = f"summary_{summary.id}_{idx}.md"
                filepath = output_dir / filename
                
                # Write content to markdown file
                with open(filepath, 'w', encoding='utf-8') as f:
                    # Write only the summary content
                    f.write(summary.summary_content or "")
                
                success_count += 1
                print(f"✓ Exported {idx}/{len(summaries)}: {filename}")
                
            except Exception as e:
                print(f"✗ Error exporting summary {idx}: {str(e)}")
        
        print(f"\n✓ Successfully exported {success_count}/{len(summaries)} files")
        print(f"✓ Files saved to: {output_dir.absolute()}")
        
    except Exception as e:
        print(f"Error connecting to database: {str(e)}")
        raise
    finally:
        db.close()

if __name__ == "__main__":
    export_summaries_to_markdown()
