"""
Example Skills - deep-research-agent

This file demonstrates how to create reusable skill functions using the new decorator pattern.
Skills are utility functions that provide practical capabilities for your agent.

NOTE: This template is for reference only. In the new decorator-based approach,
skills are defined directly in main.py using the @app.skill() decorator.

Created: 2025-07-09 16:20:06 EDT
Author: Brain Research Team
"""

from typing import Dict, Any, List, Optional
import logging
import json
import re
import time

logger = logging.getLogger(__name__)

# RECOMMENDED: Use the decorator pattern in main.py instead:
#
# @app.skill(tags=["utility"])
# def get_timestamp() -> dict:
#     """Returns the current timestamp."""
#     return {
#         "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
#         "unix_time": int(time.time()),
#     }
#
# @app.skill(tags=["text", "utility"])
# def format_text(text: str, operation: str = "title") -> dict:
#     """Simple text formatting utility."""
#     try:
#         if operation == "title":
#             formatted = text.title()
#         elif operation == "upper":
#             formatted = text.upper()
#         elif operation == "lower":
#             formatted = text.lower()
#         elif operation == "clean":
#             formatted = re.sub(r'\s+', ' ', text.strip())
#         else:
#             return {
#                 "success": False,
#                 "error": f"Unknown operation: {operation}",
#                 "available_operations": ["title", "upper", "lower", "clean"]
#             }
#         
#         return {
#             "success": True,
#             "original": text,
#             "formatted": formatted,
#             "operation": operation,
#             "length_change": len(formatted) - len(text)
#         }
#     except Exception as e:
#         return {
#             "success": False,
#             "error": str(e),
#             "original": text
#         }
#
# @app.skill(tags=["data", "json"])
# def parse_json(json_string: str) -> dict:
#     """Parse and validate JSON data."""
#     try:
#         data = json.loads(json_string)
#         return {
#             "success": True,
#             "data": data,
#             "type": type(data).__name__,
#             "size": len(str(data))
#         }
#     except json.JSONDecodeError as e:
#         return {
#             "success": False,
#             "error": f"Invalid JSON: {str(e)}",
#             "position": e.pos if hasattr(e, 'pos') else None
#         }

# Legacy examples below (for reference only)

def example_utility(data: Dict[str, Any], operation: str = "format") -> Dict[str, Any]:
    """
    DEPRECATED: Example utility function using old manual registration approach.
    
    For new projects, define skills directly in main.py using @app.skill() decorator.
    This provides automatic schema generation and better integration with the Brain system.
    
    Args:
        data (Dict[str, Any]): Input data to process
        operation (str): Type of operation to perform
        
    Returns:
        Dict[str, Any]: Processed results
    """
    
    logger.warning("Using deprecated manual skill registration. Consider migrating to @app.skill() decorator pattern.")
    logger.info(f"Processing utility operation: {operation}")
    
    try:
        if operation == "format":
            # Example formatting operation
            formatted_data = {}
            for key, value in data.items():
                if isinstance(value, str):
                    formatted_data[key] = value.title()
                else:
                    formatted_data[key] = value
                    
            return {
                "success": True,
                "operation": operation,
                "formatted": formatted_data,
                "original_keys": list(data.keys()),
                "migration_note": "Consider using @app.skill() decorator in main.py"
            }
            
        elif operation == "validate":
            # Example validation operation
            valid_keys = ["name", "email", "age"]
            missing_keys = [key for key in valid_keys if key not in data]
            extra_keys = [key for key in data.keys() if key not in valid_keys]
            
            return {
                "success": len(missing_keys) == 0,
                "operation": operation,
                "missing_keys": missing_keys,
                "extra_keys": extra_keys,
                "is_valid": len(missing_keys) == 0 and len(extra_keys) == 0,
                "migration_note": "Consider using @app.skill() decorator in main.py"
            }
            
        else:
            return {
                "success": False,
                "operation": operation,
                "error": f"Unknown operation: {operation}",
                "available_operations": ["format", "validate"],
                "migration_note": "Consider using @app.skill() decorator in main.py"
            }
            
    except Exception as e:
        logger.error(f"Error in example_utility: {str(e)}")
        return {
            "success": False,
            "operation": operation,
            "error": str(e),
            "migration_note": "Consider using @app.skill() decorator in main.py"
        }

def helper_function(text: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    DEPRECATED: Example helper function using old manual registration approach.
    
    For new projects, define this in main.py using @app.skill() decorator.
    
    Modern approach:
    
    @app.skill(tags=["text", "utility"])
    def process_text(text: str, clean_whitespace: bool = True, lowercase: bool = False) -> dict:
        try:
            processed = text
            operations = []
            
            if clean_whitespace:
                processed = re.sub(r'\s+', ' ', processed.strip())
                operations.append("clean_whitespace")
            
            if lowercase:
                processed = processed.lower()
                operations.append("lowercase")
            
            return {
                "success": True,
                "original": text,
                "processed": processed,
                "operations": operations,
                "length_change": len(processed) - len(text)
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    Args:
        text (str): Input text to process
        options (Optional[Dict[str, Any]]): Processing options
        
    Returns:
        Dict[str, Any]: Processing results
    """
    
    logger.warning("Using deprecated manual skill registration. Consider migrating to @app.skill() decorator pattern.")
    logger.info("Processing text with helper function...")
    
    if options is None:
        options = {}
    
    try:
        results = {
            "original_text": text,
            "processed_text": text,
            "statistics": {},
            "operations_applied": []
        }
        
        # Clean whitespace (default operation)
        if options.get("clean_whitespace", True):
            results["processed_text"] = re.sub(r'\s+', ' ', results["processed_text"].strip())
            results["operations_applied"].append("clean_whitespace")
        
        # Convert to lowercase
        if options.get("lowercase", False):
            results["processed_text"] = results["processed_text"].lower()
            results["operations_applied"].append("lowercase")
        
        # Remove special characters
        if options.get("remove_special", False):
            results["processed_text"] = re.sub(r'[^a-zA-Z0-9\s]', '', results["processed_text"])
            results["operations_applied"].append("remove_special")
        
        # Calculate statistics
        results["statistics"] = {
            "original_length": len(text),
            "processed_length": len(results["processed_text"]),
            "word_count": len(results["processed_text"].split()),
            "character_reduction": len(text) - len(results["processed_text"])
        }
        
        return {
            "success": True,
            "results": results,
            "options_used": options,
            "migration_note": "Consider using @app.skill() decorator in main.py"
        }
        
    except Exception as e:
        logger.error(f"Error in helper_function: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "original_text": text,
            "migration_note": "Consider using @app.skill() decorator in main.py"
        }

def json_processor(json_data: str, schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    DEPRECATED: Example JSON processing skill using old approach.
    
    For new projects, define this in main.py using @app.skill() decorator.
    
    Modern approach:
    
    @app.skill(tags=["data", "json"])
    def parse_json(json_string: str, validate_schema: bool = False) -> dict:
        try:
            data = json.loads(json_string)
            result = {
                "success": True,
                "data": data,
                "type": type(data).__name__,
                "size": len(str(data))
            }
            
            if isinstance(data, dict):
                result["keys"] = list(data.keys())
                result["key_count"] = len(data)
            elif isinstance(data, list):
                result["item_count"] = len(data)
                if data:
                    result["first_item_type"] = type(data[0]).__name__
            
            return result
        except json.JSONDecodeError as e:
            return {
                "success": False,
                "error": f"Invalid JSON: {str(e)}",
                "position": e.pos if hasattr(e, 'pos') else None
            }
    
    Args:
        json_data (str): JSON string to process
        schema (Optional[Dict[str, Any]]): Expected schema for validation
        
    Returns:
        Dict[str, Any]: Processing results
    """
    
    logger.warning("Using deprecated manual skill registration. Consider migrating to @app.skill() decorator pattern.")
    logger.info("Processing JSON data...")
    
    try:
        # Parse JSON
        parsed_data = json.loads(json_data)
        
        result = {
            "success": True,
            "parsed_data": parsed_data,
            "data_type": type(parsed_data).__name__,
            "size_info": {},
            "migration_note": "Consider using @app.skill() decorator in main.py"
        }
        
        # Add size information
        if isinstance(parsed_data, dict):
            result["size_info"]["key_count"] = len(parsed_data)
            result["size_info"]["keys"] = list(parsed_data.keys())
        elif isinstance(parsed_data, list):
            result["size_info"]["item_count"] = len(parsed_data)
            if parsed_data:
                result["size_info"]["first_item_type"] = type(parsed_data[0]).__name__
        
        # Schema validation if provided
        if schema:
            validation_errors = []
            if isinstance(parsed_data, dict) and isinstance(schema, dict):
                for key, expected_type in schema.items():
                    if key not in parsed_data:
                        validation_errors.append(f"Missing required key: {key}")
                    elif not isinstance(parsed_data[key], expected_type):
                        validation_errors.append(f"Key '{key}' should be {expected_type.__name__}, got {type(parsed_data[key]).__name__}")
            
            result["schema_validation"] = {
                "valid": len(validation_errors) == 0,
                "errors": validation_errors
            }
        
        return result
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {str(e)}")
        return {
            "success": False,
            "error": f"Invalid JSON: {str(e)}",
            "error_type": "json_decode_error",
            "migration_note": "Consider using @app.skill() decorator in main.py"
        }
    except Exception as e:
        logger.error(f"Error in json_processor: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "error_type": "general_error",
            "migration_note": "Consider using @app.skill() decorator in main.py"
        }

# Additional simple skill examples for reference

def timestamp_generator() -> Dict[str, Any]:
    """
    DEPRECATED: Simple timestamp generation.
    
    Modern approach:
    
    @app.skill(tags=["utility", "time"])
    def get_timestamp() -> dict:
        return {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "unix_time": int(time.time()),
            "iso_format": time.strftime("%Y-%m-%dT%H:%M:%SZ")
        }
    """
    
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "unix_time": int(time.time()),
        "iso_format": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "migration_note": "Use @app.skill() decorator in main.py"
    }

def text_statistics(text: str) -> Dict[str, Any]:
    """
    DEPRECATED: Text analysis utility.
    
    Modern approach:
    
    @app.skill(tags=["text", "analysis"])
    def analyze_text(text: str) -> dict:
        words = text.split()
        sentences = text.split('.')
        
        return {
            "character_count": len(text),
            "word_count": len(words),
            "sentence_count": len([s for s in sentences if s.strip()]),
            "average_word_length": sum(len(word) for word in words) / len(words) if words else 0,
            "longest_word": max(words, key=len) if words else "",
            "shortest_word": min(words, key=len) if words else ""
        }
    """
    
    words = text.split()
    sentences = text.split('.')
    
    return {
        "character_count": len(text),
        "word_count": len(words),
        "sentence_count": len([s for s in sentences if s.strip()]),
        "average_word_length": sum(len(word) for word in words) / len(words) if words else 0,
        "longest_word": max(words, key=len) if words else "",
        "shortest_word": min(words, key=len) if words else "",
        "migration_note": "Use @app.skill() decorator in main.py"
    }

def url_validator(url: str) -> Dict[str, Any]:
    """
    DEPRECATED: URL validation utility.
    
    Modern approach:
    
    @app.skill(tags=["validation", "url"])
    def validate_url(url: str) -> dict:
        import re
        
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        
        is_valid = bool(url_pattern.match(url))
        
        return {
            "url": url,
            "is_valid": is_valid,
            "protocol": url.split("://")[0] if "://" in url else None,
            "has_port": ":" in url.split("://")[1] if "://" in url and len(url.split("://")) > 1 else False
        }
    """
    
    import re
    
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    is_valid = bool(url_pattern.match(url))
    
    return {
        "url": url,
        "is_valid": is_valid,
        "protocol": url.split("://")[0] if "://" in url else None,
        "has_port": ":" in url.split("://")[1] if "://" in url and len(url.split("://")) > 1 else False,
        "migration_note": "Use @app.skill() decorator in main.py"
    }
