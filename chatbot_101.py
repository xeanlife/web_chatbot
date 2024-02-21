import sys
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import tkinter as tk
from tkinter import filedialog


def get_pdf_text():
	file_path = filedialog.askopenfilenames()
	print(file_path)
	text = ''
 
	for pdf in file_path:
		pdf_reader = PdfReader(pdf)
		for page in pdf_reader.pages:
			text += page.extract_text()
	return text


def get_text_chunks(text):
	text_splitter = CharacterTextSplitter(separator='\n', chunk_size=500, chunk_overlap=100, length_function=len)
	chunks = text_splitter.split_text(text)
	return chunks


def get_vectorstore(text_chunks, key):
	embeddings = OpenAIEmbeddings(openai_api_key=key)
	vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
	return vectorstore


def get_conversation_chain(vectorstore, key):
	llm = ChatOpenAI(openai_api_key=key)
	memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
	conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(search_type='similarity', search_kwargs={'k': 4}), memory=memory)
	return conversation_chain


def handle_userinput(user_question, conversation):
	response = conversation({'question': user_question})
	return response


def main():
	print('-------------------------------------------------------')
	print('Welcome to ChatBot 101!')
	print('-------------------------------------------------------\n')
	key = input('Enter your openAI API key: ')
	print('Processing ...')

	# key = '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'
	# pdf_address = 'Ads_cookbook.pdf'

	try:
		raw_text = get_pdf_text()
		text_chunks = get_text_chunks(raw_text)
		vectorstore = get_vectorstore(text_chunks, key)
		conversation = get_conversation_chain(vectorstore, key)
	except:
		print('\nInvalid Inputs, Try Again')
		print('-------------------------------------------------------')
		sys.exit()

	user_question = ''
	while True:
		user_question = input('\nSay something (exit to quit): ')
		if user_question != 'exit':
			response = handle_userinput(user_question, conversation)
			print(response['answer'])
		else:
			break
  
	print('\nQuitting, thanks for using ChatBot 101!')
	print('-------------------------------------------------------')


if __name__ == '__main__':
  root = tk.Tk()
  root.withdraw()
  main()