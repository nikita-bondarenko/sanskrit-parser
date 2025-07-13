"""
API Routes for Sanskrit OCR application
"""

from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse
from PIL import Image
import io
import logging

from auth import LoginRequest, LoginResponse, verify_admin_password, create_admin_token, verify_admin_token
from ocr_service import ocr_service
from sanskrit_database import sanskrit_db, TextMatch
from text_extractor import text_extractor

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

@router.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Sanskrit OCR API is running"}

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": True}

@router.post("/admin/login")
async def admin_login(request: LoginRequest):
    """Admin login endpoint"""
    try:
        if verify_admin_password(request.password):
            token = create_admin_token()
            return LoginResponse(
                success=True,
                token=token,
                message="Login successful"
            )
        else:
            return LoginResponse(
                success=False,
                message="Invalid password"
            )
    except Exception as e:
        logger.error(f"Admin login error: {e}")
        return LoginResponse(
            success=False,
            message="Login failed"
        )

@router.post("/ocr")
async def ocr_endpoint(file: UploadFile = File(...)):
    """OCR endpoint for Sanskrit text recognition with database matching"""
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        logger.info(f"Processing image: {image.size}, mode: {image.mode}")
        
        # Process image with OCR service
        ocr_result = ocr_service.process_image(image)
        
        # Try to find a match in the database
        database_match = sanskrit_db.find_best_match(ocr_result['text'], min_confidence=0.7)
        
        # Use database match if found and confidence is high enough
        final_text = ocr_result['text']
        source_info = None
        
        if database_match and database_match.confidence > 0.8:
            final_text = database_match.matched_text
            source_info = {
                "source_book": database_match.source_book,
                "source_chapter": database_match.source_chapter,
                "source_verse": database_match.source_verse,
                "confidence": database_match.confidence,
                "match_type": database_match.match_type
            }
            logger.info(f"Database match found: {database_match.source_book} (confidence: {database_match.confidence:.2f})")
        
        # Log recognition metrics
        log_recognition_metrics(image.size, ocr_result['stats'], final_text)
        
        return JSONResponse({
            "success": True,
            "text": final_text,
            "original_ocr": ocr_result['text'],
            "source_info": source_info,
            "image_info": {
                "width": image.size[0],
                "height": image.size[1],
                "mode": image.mode,
                "stats": ocr_result['stats']
            },
            "processing_info": {
                **ocr_result['processing_info'],
                "database_match": database_match is not None,
                "quality_score": ocr_result['quality_score']
            }
        })
        
    except Exception as e:
        logger.error(f"OCR error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")

@router.post("/upload-book")
async def upload_book(file: UploadFile = File(...), admin_verified: bool = Depends(verify_admin_token)):
    """Upload and process a book file to add to the Sanskrit database"""
    try:
        # Validate file type
        supported_formats = text_extractor.get_supported_formats()
        file_extension = file.filename.split('.')[-1].lower()
        
        if f'.{file_extension}' not in supported_formats:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file format. Supported: {', '.join(supported_formats)}"
            )
        
        # Read file content
        file_content = await file.read()
        
        # Extract texts from file
        sanskrit_texts = text_extractor.process_file(file_content, file.filename)
        
        if not sanskrit_texts:
            raise HTTPException(status_code=400, detail="No text could be extracted from the file")
        
        # Add texts to database
        added_count = 0
        for sanskrit_text in sanskrit_texts:
            try:
                sanskrit_db.add_text(sanskrit_text)
                added_count += 1
            except Exception as e:
                logger.error(f"Error adding text to database: {e}")
        
        # Get updated database statistics
        stats = sanskrit_db.get_statistics()
        
        return JSONResponse({
            "success": True,
            "message": f"Successfully processed {file.filename}",
            "texts_added": added_count,
            "total_texts_extracted": len(sanskrit_texts),
            "database_stats": stats
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Book upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Book processing failed: {str(e)}")

@router.get("/database-stats")
async def get_database_stats():
    """Get current database statistics"""
    try:
        stats = sanskrit_db.get_statistics()
        return JSONResponse({
            "success": True,
            "stats": stats
        })
    except Exception as e:
        logger.error(f"Database stats error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get database stats: {str(e)}")

@router.get("/search-books")
async def search_books(book_name: str):
    """Search for texts by book name"""
    try:
        texts = sanskrit_db.search_by_source(book_name)
        
        # Convert to serializable format
        results = []
        for text in texts:
            results.append({
                "id": text.id,
                "text": text.text[:200] + "..." if len(text.text) > 200 else text.text,
                "source_book": text.source_book,
                "source_chapter": text.source_chapter,
                "source_verse": text.source_verse,
                "text_type": text.text_type,
                "language_script": text.language_script,
                "word_count": text.word_count
            })
        
        return JSONResponse({
            "success": True,
            "results": results,
            "total_found": len(results)
        })
        
    except Exception as e:
        logger.error(f"Book search error: {e}")
        raise HTTPException(status_code=500, detail=f"Book search failed: {str(e)}")

@router.get("/supported-formats")
async def get_supported_formats():
    """Get list of supported file formats for book upload"""
    try:
        formats = text_extractor.get_supported_formats()
        return JSONResponse({
            "success": True,
            "supported_formats": formats
        })
    except Exception as e:
        logger.error(f"Supported formats error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get supported formats: {str(e)}")

def log_recognition_metrics(image_size: tuple, stats: dict, final_text: str):
    """Log metrics for accuracy analysis"""
    
    metrics = {
        'image_width': image_size[0],
        'image_height': image_size[1],
        'aspect_ratio': image_size[0] / image_size[1],
        'estimated_lines': stats.get('line_count', 0),
        'text_density': stats.get('text_density', 0),
        'character_density': stats.get('character_density', 0),
        'output_length': len(final_text),
        'output_lines': len(final_text.split('\n')),
        'contains_diacritics': bool(set('а̄ӣӯр̣л̣н̣т̣д̣ш́ṃ') & set(final_text))
    }
    
    logger.info(f"Recognition metrics: {metrics}")
    
    # Log potential quality indicators
    quality_score = ocr_service.calculate_quality_score(final_text, stats)
    logger.info(f"Estimated quality score: {quality_score:.2f}/10") 