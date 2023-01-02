import requests
from bs4 import BeautifulSoup
import pandas as pd
'''links = []
for i in range(1,86,1):
    print(i)
    response = requests.get(f'https://g1.globo.com/economia/dolar/index/feed/pagina-{i}.ghtml')
    content = response.content
    site = BeautifulSoup(content, 'html.parser')
    #HTML da notícia
    noticias = site.findAll('div', attrs={'class': 'feed-post-body'})

    for noticia in noticias:
        titulo = noticia.find('a', attrs={'class': 'feed-post-link'})
        link = titulo['href']
        links.append(link)

ll = pd.DataFrame(links)
ll.to_csv('links_dolar_g1.csv', index='links')'''

links = pd.read_csv('links_dolar_g1.csv')['links'].tolist()

conteudos = []
datas = []
for link in links:
    print(link)
    response = requests.get(link)
    content = response.content
    site = BeautifulSoup(content, 'html.parser')
    textos = [a.text for a in site.findAll('p', attrs={'class': 'content-text__container'})]
    if len(' '.join(textos)) > 10:
        conteudos.append(' '.join(textos))
        data_publicacao = site.find('time', attrs={'itemprop': 'datePublished'})
        datas.append(((data_publicacao.text).split(' ')[1]).replace('/', '.'))


df = pd.DataFrame()
df['Data'] = datas
df['Textos'] = conteudos

df.to_excel('teste.xlsx', index=None)


