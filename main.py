from src.train import (load_data, features, train_split, build_pipeline, train_model, save_model)

if __name__ == "__main__":
    
    #1. Load
    credit = load_data()
    
    #2. Features
    num_credit = features(credit)
    
    #3. Split
    X_train, X_test, y_train, y_test = train_split(num_credit)
    
    #4. Build Pipline 
    pipeline = build_pipeline()
    
    #5. Train
    pipeline = train_model(pipeline, X_train, y_train)
    
    #6. Save
    save_model(pipeline)
    
    print("Done — model saved to models/model.pkl")