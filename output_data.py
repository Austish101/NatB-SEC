import requests
api_url = ""
port =8081
def API_post(port, error_type, 
             team, alive_count, not_alive_count, 
             lost_reason, missed_total_count, tag):
    
    myData={"team" : team, 
            "error_type" : error_type,
            "alive_count" : alive_count, 
            "not_alive_count" : not_alive_count, 
            "lost_reason" : lost_reason,
            "missed_total_count" : missed_total_count}
    
    requests.post(api_url, json=myData)
