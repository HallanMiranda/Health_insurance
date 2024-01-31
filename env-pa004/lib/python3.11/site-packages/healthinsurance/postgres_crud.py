import psycopg2
import pandas as pd
import numpy as np

class Postgresql:

    def create_table(self, query,user, password, host, port, database, connect = True):

        try:
            connection = psycopg2.connect(user = user ,
                                    password = password ,
                                    host = host ,
                                    port = port ,
                                    database = database )

            cursor = connection.cursor()

            create_query = query
            cursor.execute(create_query)
            connection.commit()
            print("Tabela criada no postgresql com sucesso ")

        except (Exception, psycopg2.DatabaseError) as error :
            print ("Erro durante a criação da tabela ", error)

        finally:

                if connect:
                    cursor.close()
                    connection.close()
                    print("Conexão com Postgresql fechada")

    def retrieve_data(query, user, password, host, port, database, connect = True, objeto = 'pd'):

        try:
            connection = psycopg2.connect(user = user ,
                                    password = password ,
                                    host = host ,
                                    port = port ,
                                    database = database )

            cursor = connection.cursor()

            select_query = query

            cursor.execute(select_query)
            print("Buscando os dados!!!")

            resultado = cursor.fetchall()
            colunas = [desc[0] for desc in cursor.description]

            if objeto == 'pd':
                base = pd.DataFrame(resultado, columns=colunas)
            else:
                base = np.array(resultado)
            return base

        except (Exception, psycopg2.DatabaseError) as error :
            print ("Erro durante a querry", error)

        finally:

                if connect:
                    cursor.close()
                    connection.close()
                    print("Conexão com Postgresql fechada")


    def insert_data(self, df, tabela, user, password, host, port, database, connect = True):

        try:

            connection = psycopg2.connect(user = user ,
                                    password = password ,
                                    host = host ,
                                    port = port ,
                                    database = database )
            cursor = connection.cursor()


            cols=",".join([str(i) for i in df.columns.tolist()])

            for _,row in df.iterrows():
                sql = "INSERT INTO" + tabela + "(" +cols + ") VALUES (" + "%s,"*(len(row)-1) + "%s)"
                cursor.execute(sql, tuple(row))
                connection.commit()

        except (Exception, psycopg2.Error) as error :
            if connection:
                print("Erro durante a criação da tabela ", error)

        finally:

            if connect:
                cursor.close()
                connection.close()
                print("Conexão com Postgresql fechada")