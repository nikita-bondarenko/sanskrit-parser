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
from converter import text_converter

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
async def ocr_endpoint(
    file: UploadFile = File(...),
    output_format: str = "iast",
    input_format: str = "english_diacritics"
):
    """OCR endpoint for Sanskrit text recognition with database matching and format conversion"""
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Validate output format
        if output_format not in ['iast', 'russian']:
            raise HTTPException(status_code=400, detail="output_format must be 'iast' or 'russian'")
        
        # Validate input format
        valid_input_formats = ['russian_diacritics', 'english_diacritics']
        if input_format not in valid_input_formats:
            raise HTTPException(status_code=400, detail=f"input_format must be one of: {', '.join(valid_input_formats)}")
        
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        logger.info(f"Processing image: {image.size}, mode: {image.mode}, input_format: {input_format}, output_format: {output_format}")
        
        # Process image with OCR service including input format
        ocr_result = ocr_service.process_image(image, input_format)
        
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
        
        # Convert to target format based on input format specified by user
        # Map input format names to converter format names
        format_mapping = {
            'english_diacritics': 'iast',
            'russian_diacritics': 'russian'
        }
        
        source_format = format_mapping.get(input_format, 'iast')
        target_format = output_format
        
        logger.info(f"Converting text from '{source_format}' (input_format: {input_format}) to '{target_format}' (output_format: {output_format})")
        logger.info(f"Text sample for conversion: '{final_text[:50]}...'")
        
        if source_format != target_format:
            converted_text = text_converter.convert_text_format(final_text, target_format, source_format)
            logger.info(f"Text converted from '{source_format}' to '{target_format}': '{converted_text[:50]}...'")
        else:
            converted_text = final_text
            logger.info(f"No conversion needed, both formats are '{target_format}'")
        
        # Log recognition metrics
        log_recognition_metrics(image.size, ocr_result['stats'], converted_text)
        
        return JSONResponse({
            "success": True,
            "text": converted_text,
            "original_ocr": ocr_result['text'],
            "source_info": source_info,
            "image_info": {
                "width": image.size[0],
                "height": image.size[1],
                "mode": image.mode,
                "stats": ocr_result['stats']
            },
            "processing_info": {
                **ocr_result.get('processing_info', {}),
                "database_match": database_match is not None,
                "quality_score": ocr_result.get('quality_score', 0.0),
                "format_converted": source_format != target_format,
                "source_format": source_format,
                "output_format": output_format,
                "input_format": input_format,  # Ensure input_format is included
                "processing_method": ocr_result.get('processing_method', 'pytorch')
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

@router.post("/convert-text")
async def convert_text_format_endpoint(request: dict):
    """Convert text between IAST and Russian diacritics formats"""
    try:
        text = request.get('text', '')
        target_format = request.get('target_format', 'russian')
        source_format = request.get('source_format')  # Optional, will auto-detect if None
        
        if not text:
            raise HTTPException(status_code=400, detail="Text is required")
        
        if target_format not in ['iast', 'russian']:
            raise HTTPException(status_code=400, detail="target_format must be 'iast' or 'russian'")
        
        # Auto-detect source format if not provided
        detected_format = text_converter.detect_text_format(text)
        actual_source_format = source_format or detected_format
        
        # Convert text
        converted_text = text_converter.convert_text_format(text, target_format, actual_source_format)
        
        return JSONResponse({
            "success": True,
            "original_text": text,
            "converted_text": converted_text,
            "source_format": actual_source_format,
            "target_format": target_format,
            "detected_format": detected_format
        })
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Text conversion error: {e}")
        raise HTTPException(status_code=500, detail=f"Text conversion failed: {str(e)}")

@router.post("/process-text")
async def process_text_endpoint(request: dict):
    """Process text input (without OCR) with database matching and format conversion"""
    try:
        text = request.get('text', '')
        output_format = request.get('output_format', 'russian')
        
        if not text:
            raise HTTPException(status_code=400, detail="Text is required")
        
        if output_format not in ['iast', 'russian']:
            raise HTTPException(status_code=400, detail="output_format must be 'iast' or 'russian'")
        
        logger.info(f"Processing text input: {text[:50]}...")
        
        # Try to find a match in the database
        database_match = sanskrit_db.find_best_match(text, min_confidence=0.7)
        
        # Use database match if found and confidence is high enough
        final_text = text
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
        
        # Convert to target format if needed
        current_format = text_converter.detect_text_format(final_text)
        if current_format != output_format:
            converted_text = text_converter.convert_text_format(final_text, output_format, current_format)
        else:
            converted_text = final_text
        
        return JSONResponse({
            "success": True,
            "text": converted_text,
            "original_text": text,
            "source_info": source_info,
            "text_info": {
                "word_count": len(final_text.split()) if final_text else 0,
                "character_count": len(final_text) if final_text else 0,
                "detected_format": current_format,
                "output_format": output_format
            },
            "processing_info": {
                "database_match": database_match is not None,
                "format_converted": current_format != output_format,
                "input_type": "text"
            }
        })
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Text processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Text processing failed: {str(e)}")

@router.get("/test-macron/{char}")
async def test_macron_recognition(char: str = "a"):
    """Test macron recognition on a character"""
    try:
        # Create test image with macron
        test_image = ocr_service.create_test_macron_image(char)
        
        # Test both stages
        base_char, confidence = ocr_service.recognize_base_character(test_image)
        diacritics = ocr_service.detect_diacritics_advanced(test_image)
        final_char = ocr_service.combine_char_with_diacritics(base_char, diacritics)
        
        return {
            "test_char": char,
            "recognized_base": base_char,
            "confidence": confidence,
            "detected_diacritics": diacritics,
            "final_character": final_char,
            "test_type": "macron",
            "pipeline": "two_stage_character_recognition"
        }
        
    except Exception as e:
        logger.error(f"Macron test failed: {e}")
        raise HTTPException(status_code=500, detail=f"Macron test failed: {str(e)}")

@router.get("/api/test-pytorch-char/{char}")
async def test_pytorch_character_recognition(char: str, diacritics: str = ""):
    """Test PyTorch character recognition with optional diacritics"""
    try:
        diacritic_list = diacritics.split(",") if diacritics else []
        diacritic_list = [d.strip() for d in diacritic_list if d.strip()]
        
        result = ocr_service.test_character_recognition(char, diacritic_list)
        
        return {
            "status": "success",
            "result": result,
            "pipeline": "pytorch_two_stage"
        }
        
    except Exception as e:
        logger.error(f"PyTorch character test failed: {e}")
        raise HTTPException(status_code=500, detail=f"PyTorch test failed: {str(e)}")

@router.get("/api/test-pytorch-text/{text}")
async def test_pytorch_text_recognition(text: str):
    """Test PyTorch text recognition by creating synthetic text image"""
    try:
        # Create a simple text image
        from PIL import Image, ImageDraw, ImageFont
        
        img = Image.new('L', (400, 100), color=255)
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()
        
        # Draw text
        draw.text((10, 30), text, fill=0, font=font)
        
        # Process with PyTorch OCR
        result = ocr_service.process_image(img)
        
        return {
            "status": "success",
            "input_text": text,
            "recognized_text": result.get('text', ''),
            "confidence": result.get('confidence', 0.0),
            "processing_method": result.get('processing_method', 'unknown'),
            "stats": result.get('stats', {}),
            "pipeline": "pytorch_full_text"
        }
        
    except Exception as e:
        logger.error(f"PyTorch text test failed: {e}")
        raise HTTPException(status_code=500, detail=f"PyTorch text test failed: {str(e)}")

@router.get("/api/test-diacritics")
async def test_all_diacritics():
    """Test all diacritic types with different base characters"""
    try:
        test_cases = [
            ("a", ["macron"]),           # ā
            ("r", ["dot_below"]),        # ṛ  
            ("s", ["dot_below"]),        # ṣ
            ("s", ["acute"]),            # ś
            ("m", ["dot_above"]),        # ṃ
            ("h", ["visarga"]),          # ḥ
            ("n", ["dot_below"]),        # ṇ
            ("t", ["dot_below"]),        # ṭ
            ("d", ["dot_below"]),        # ḍ
            ("i", ["macron"]),           # ī
            ("u", ["macron"]),           # ū
        ]
        
        results = []
        for char, diacritics in test_cases:
            try:
                result = ocr_service.test_character_recognition(char, diacritics)
                results.append(result)
            except Exception as e:
                results.append({
                    "input_char": char,
                    "input_diacritics": diacritics,
                    "error": str(e),
                    "success": False
                })
        
        successful = sum(1 for r in results if r.get('success', False))
        
        return {
            "status": "success",
            "total_tests": len(test_cases),
            "successful": successful,
            "success_rate": successful / len(test_cases),
            "results": results,
            "pipeline": "pytorch_diacritic_comprehensive"
        }
        
    except Exception as e:
        logger.error(f"Comprehensive diacritic test failed: {e}")
        raise HTTPException(status_code=500, detail=f"Comprehensive test failed: {str(e)}")

@router.get("/api/pytorch-info")
async def pytorch_info():
    """Get information about PyTorch OCR setup"""
    try:
        import torch
        
        info = {
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": str(ocr_service.pytorch_service.device),
            "base_characters": len(ocr_service.pytorch_service.char_to_idx),
            "diacritic_types": [
                "macron (long vowels)",
                "dot_below (retroflexes)", 
                "dot_above (anusvara)",
                "acute (palatals)",
                "visarga (aspiration)"
            ],
            "model_status": {
                "base_char_model": "initialized",
                "diacritic_model": "initialized"
            },
            "pipeline": "two_stage_pytorch_recognition"
        }
        
        return {
            "status": "success",
            "info": info
        }
        
    except Exception as e:
        logger.error(f"PyTorch info failed: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

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