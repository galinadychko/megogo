def more_than_a_half(row):
    if row["watched_time"] >= row["duration"]/2:
        return_value = True
    else:
        return_value = False
    return return_value
        
