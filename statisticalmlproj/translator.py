#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:



import torch

# List available models
torch.hub.list('pytorch/fairseq')  # [..., 'transformer.wmt16.en-de', ... ]

de2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.de-en.single_model', tokenizer='moses', bpe='fastbpe')
# Load a transformer trained on WMT'16 En-De
#en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt16.en-de', tokenizer='moses', bpe='subword_nmt')
#en2de.eval()  # disable dropout
de2en.eval()
# The underlying model is available under the *models* attribute
#assert isinstance(en2de.models[0], fairseq.models.transformer.TransformerModel)


# Move model to GPU for faster translation
de2en.cpu()




# Batched translation
de2en.translate(['Hallo'])


# In[2]:


from flask import Flask, request, render_template
import requests
import torch

url = 'http://localhost:3000/predict'

app = Flask(__name__, template_folder="templates")

# Load the model

@app.route('/', methods=['GET','POST'])
def home():
    if request.method == 'POST':
        data = request.form.get('text1')
        answer=de2en.translate(data)
       
        
        return render_template('demo2.html', sentiment=answer)
      
           
            
    return render_template('demo2.html', sentiment='')
        
        
    
if __name__ == '__main__':
    app.run(port=3000, debug=False)


# In[ ]:




