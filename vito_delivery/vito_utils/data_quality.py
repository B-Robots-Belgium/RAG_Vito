import great_expectations as gx
import pandas as pd
import os

connection_string = f"postgresql+psycopg2://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DATABASE')}"
data_source_name = "vito_datasource"

data_context = gx.get_context()

data_source = data_context.data_sources.add_postgres(
    name=data_source_name, connection_string=connection_string
)



if __name__ == "__main__":
    print("Running file as main file. Testing API calls...")
    print(10 * '-')
    asset_name = "VITO_ASSET"
    database_table_name = "vito_article"
    table_data_asset = data_source.add_table_asset(
        table_name=database_table_name, name=asset_name
    )

    asset_name = "QUERY_ASSET"
    asset_query = "SELECT * from vito_article"
    query_data_asset = data_source.add_query_asset(query=asset_query, name=asset_name)
    full_table_batch_definition = table_data_asset.add_batch_definition_whole_table(
    name="FULL_TABLE"
    )
    full_table_batch = full_table_batch_definition.get_batch()

    keyword_batch_definition = table_data_asset.add_batch_definition(

    )
    
    print(full_table_batch.head())

