import sklearn
import joblib
import json

# entrypoint
def handler(event, context):
    
    print(event)
    json_event = event

    data = [
        json_event.get("yr"),
        json_event.get("mo"),
        json_event.get("day"),
        json_event.get("loc_WC1"),
        json_event.get("loc_bathroom1"),
        json_event.get("loc_bedroom1"),
        json_event.get("loc_conservatory"),
        json_event.get("loc_dining_room"),
        json_event.get("loc_hallway"),
        json_event.get("loc_kitchen"),
        json_event.get("loc_living_room"),
        json_event.get("loc_lounge"),
        json_event.get("loc_study"),
        json_event.get("norm_time")
    ]
    
    if all(item is not None for item in data):
        loaded_model = joblib.load('tenancy_logreg_model.pkl')
        predict = loaded_model.predict([data])[0]
        print(f"PREDICT = {predict}")
        return {
                'statusCode': 200,
                'body': json.loads(json.dumps(predict, default=str)),
                'headers': {
                    'Access-Control-Allow-Origin': '*', 
                    'Access-Control-Allow-Credentials': True, 
                    'Content-Type': 'application/json',
                },
            }

    else:
        # Handle the case where one or more items are missing
        return "Required Values Missing. please view https://github.com/MegaBytten/UKDRI-DataEng-Model/blob/main/analysis.ipynb for more info."
