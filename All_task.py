class Data_Handler:

    def __init__(self) -> dict:
        import yaml
        config_yaml_file='config.yaml'
        with open(config_yaml_file, 'r') as file:
            self.config = yaml.safe_load(file)

    def load_llm(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig,pipeline
        from langchain.llms import HuggingFacePipeline 
        model_id = self.config['model_config']['model']
        bnb_config = BitsAndBytesConfig \
                    (
                        load_in_4bit=True,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16,
                    )
        model = AutoModelForCausalLM.from_pretrained (model_id, trust_remote_code=True,
                                                    quantization_config=bnb_config,
                                                    device_map="cuda")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        text_generation_pipeline = pipeline(
            model=model,
            tokenizer=tokenizer,
            task="text-generation",
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
            return_full_text=False,
            max_new_tokens=300,
            temperature = 0.3,
            do_sample=True,
        )
        llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
        return llm

    def text_splitter_fun(self,input_text):
        from langchain.text_splitter import CharacterTextSplitter
        text_splitter = CharacterTextSplitter(        
            separator = "\n",
            chunk_size = self.config['text_split_config']['chunk_size'],
            chunk_overlap  = self.config['text_split_config']['chunk_overlap'],
            length_function = len,
        )
        splitted_text = text_splitter.split_text(input_text)
        return splitted_text

    
    def get_vectorstore(self,input_text):
        from langchain.vectorstores import FAISS
        from langchain_community.embeddings import HuggingFaceBgeEmbeddings
        model_name = self.config['model_config']['embedding_model']
        model_kwargs = {"device": "cpu"}
        encode_kwargs = {"normalize_embeddings": True}
        hf = HuggingFaceBgeEmbeddings(
            model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
        )
        vectorstore = FAISS.from_texts(input_text, hf)
        return vectorstore

class Tasks(Data_Handler):

    def __init__(self,text):
        super().__init__()
        self.text=text
        self.splitted_text=self.text_splitter_fun(self.text)
        self.llm=self.load_llm()
        self.ve_store=self.get_vectorstore(self.text)
        
    def summarize_text(self,prompt_style):
        from langchain.schema.document import Document
        from langchain.prompts import PromptTemplate
        from langchain.chains.llm import LLMChain
        from langchain.chains.combine_documents.stuff import StuffDocumentsChain
        #Converting text to LangChain documents so that StuffDocumentsChain can understand Input
        documents = Document(page_content=self.text, metadata={"source": "local"})


        # Define prompt with prompt template
        prompt_template = prompt_style+ """Write a concise summary of the following: 
        "{text}"
        CONCISE SUMMARY"""
        prompt = PromptTemplate.from_template(prompt_template)

        # Define LLM chain
        llm_chain = LLMChain(llm=self.load_llm(), prompt=prompt)

        # Define StuffDocumentsChain
        stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")

        #Get the Summary of text by invoking StuffDocumentsChain
        summary = stuff_chain.invoke([documents])
        return summary['output_text']

    def qa_task(self,question,prompt_style):
        from langchain import PromptTemplate
        from langchain.chains import RetrievalQA

        DEFAULT_SYSTEM_PROMPT = """
        You are a helpful, respectful and honest assistant. Always answer as helpfully
        as possible, while being safe. Your answers should not include any harmful,
        unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your
        responses are socially unbiased and positive in nature.
        
        If a question does not make any sense, or is not factually coherent, explain
        why instead of answering something not correct. If you don't know the answer to a
        question, please don't share false information.
        """.strip()
 
        def generate_prompt(prompt: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
            return f"""
        [INST] <<SYS>>
        {system_prompt}
        <</SYS>>
        
        {prompt} [/INST]
        """.strip()

        SYSTEM_PROMPT = "Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer."

        template = generate_prompt(
                """{context}
            Question: {question}
            """,
                system_prompt=SYSTEM_PROMPT,
        ) 
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])

        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.ve_store.as_retriever(search_type="similarity",search_kwargs={'k': 4}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt},
        )
        result = qa_chain(question)
        print(result['result'])
        return result['result']

if __name__=="__main__":
    text=""
    summ_obj=Tasks(text)
    summ_obj.config