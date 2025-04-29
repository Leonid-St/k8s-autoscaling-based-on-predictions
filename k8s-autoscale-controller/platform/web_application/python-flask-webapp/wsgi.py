from app import app as application
import os

if __name__ == "__main__":
    application.run(host='0.0.0.0', port=int(os.environ.get("PUBLISH_PORT", 8000)))
