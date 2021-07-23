import json
import os  
from wordpress_xmlrpc import WordPressPost, Client
from wordpress_xmlrpc.methods import posts
from logwp import site, user, password
from functions import generate_wp

client = Client(site, user, password)
with open('amazonia_hurricane_hypnosis.json') as file:
    data = json.load(file)

generate_wp(data,client)

for filename in os.listdir("outputs"):

    with open(f'outputs/{filename}') as file:
        title = file.readline()
        content = file.read()

    post = WordPressPost()
    post.title = title
    post.content = content
    
    print(posts.NewPost(post))
    post.id = client.call(posts.NewPost(post))

    post.post_status = 'publish'
    client.call(posts.EditPost(post.id, post))

os.remove("Compressed_imagename0.jpg")
os.remove("image_name.jpg")