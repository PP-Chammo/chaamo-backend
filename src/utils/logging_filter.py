import logging


class HealthCheckFilter(logging.Filter):
    """Filter to suppress health check access logs from Uvicorn."""
    
    def filter(self, record):
        # Filter out health check requests
        if hasattr(record, 'args') and record.args:
            # Check if it's an access log record with the message format
            if len(record.args) >= 3:
                method_path = record.args[1] if len(record.args) > 1 else ""
                if isinstance(method_path, str) and "GET /health" in method_path:
                    return False
        
        # Check the message itself
        if hasattr(record, 'message'):
            if "GET /health" in record.message:
                return False
        
        # Check the raw message
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            if "GET /health" in record.msg:
                return False
        
        return True
