"""
Main entry point for running backend as a module.
Usage: python -m backend
"""

from backend import app, socketio, logger
from config import Config

if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("Starting IDS Backend Server")
    logger.info("=" * 70)
    logger.info(f"REST API: http://{Config.WEBSOCKET_HOST}:{Config.WEBSOCKET_PORT}")
    logger.info(f"WebSocket: ws://{Config.WEBSOCKET_HOST}:{Config.WEBSOCKET_PORT}")
    logger.info(f"Debug Mode: {Config.DEBUG_MODE}")
    logger.info("=" * 70)
    
    try:
        # Run with SocketIO (handles both HTTP and WebSocket)
        socketio.run(
            app,
            host=Config.WEBSOCKET_HOST,
            port=Config.WEBSOCKET_PORT,
            debug=Config.DEBUG_MODE,
            use_reloader=Config.DEBUG_MODE,
            log_output=Config.DEBUG_MODE
        )
    except KeyboardInterrupt:
        logger.info("\n\nShutting down server...")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise
    finally:
        logger.info("Server stopped")