import psycopg2
from psycopg2 import sql

try:
    connection = psycopg2.connect(
        dbname="map_data", 
        user="postgres", 
        password="oddopolis", 
        host="localhost", 
        port="5432"
    )

    speed_limits = {
        'motorway': 65,
        'trunk': 55,
        'primary': 55,
        'secondary': 45,
        'tertiary': 40,
        'unclassified': 35,
        'motorway_link': 45,
        'trunk_link': 40,
        'primary_link': 35,
        'secondary_link': 35,
        'tertiary_link': 30,
        'residential': 35,
        'living_street': 15,
        'service': 10,
        'pedestrian': 5,
    }

    cursor = connection.cursor()

    for highway_type, speed_limit in speed_limits.items():
        query = sql.SQL(
            f"""
            UPDATE planet_osm_line
            SET maxspeed = %s
            WHERE highway = %s AND maxspeed IS NULL;
            """
        )
        cursor.execute(query, (speed_limit, highway_type))

    connection.commit()

except Exception as e:
    print(f"Unable to access database, error: {e}")
finally:
    print("done")
    if cursor:
        cursor.close()
    if connection:
        connection.close()

