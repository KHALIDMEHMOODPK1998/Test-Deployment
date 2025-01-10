from fastapi import FastAPI
import uvicorn

# Create a FastAPI instance
app = FastAPI()

# Define a root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to FastAPI!"}

# Define an endpoint with a query parameter
@app.get("/greet/")
def greet_user(name: str = "Guest"):
    return {"message": f"Hello, {name}!"}

# Define an endpoint with a path parameter
@app.get("/items/{item_id}")
def read_item(item_id: int, details: bool = False):
    if details:
        return {"item_id": item_id, "details": f"Details about item {item_id}"}
    return {"item_id": item_id}

# Define a POST endpoint
@app.post("/create-item/")
def create_item(item_name: str, quantity: int):
    return {"message": f"Item '{item_name}' with quantity {quantity} created successfully!"}

# Main function to run the application
def main():
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

# Run the main function if the script is executed directly
if __name__ == "__main__":
    main()
