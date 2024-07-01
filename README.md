# Dự Án Chatbot

Đây là mã nguồn cho dự án chatbot sử dụng các thư viện transformers, langchain, và langchain_huggingface.

## Yêu Cầu
Hãy đảm bảo rằng bạn đã cài đặt các thư viện sau:
     
    pip install transformers==4.41.2
    pip install bitsandbytes==0.43.1
    pip install accelerate==0.31.0
    pip install langchain==0.2.5
    pip install langchainhub==0.1.20
    pip install langchain-chroma==0.1.1
    pip install langchain-community==0.2.5
    pip install langchain_huggingface==0.0.3   
    pip install python-dotenv==1.0.1
    pip install pypdf==4.2.0
    pip install numpy==1.25.0

## Hướng Dẫn Sử Dụng
Import các thư viện cần thiết:

    import torch
    import numpy as np
    from transformers import BitsAndBytesConfig
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_huggingface.llms import HuggingFacePipeline
    from langchain.memory import ConversationBufferMemory
    from langchain_community.chat_message_histories import ChatMessageHistory
    from langchain_community.document_loaders import PyPDFLoader, TextLoader
    from langchain.chains import ConversationalRetrievalChain
    from langchain_chroma import Chroma
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
    from langchain import hub

## Tải và xử lý tài liệu:

    Loader = PyPDFLoader
    FILE_PATH = "/content/YOLOv10_Tutorials.pdf"
    loader = Loader(FILE_PATH)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    print("Number of sub-documents: ", len(docs))
    print(docs[0])

## Tạo embedding và retriever:

    embedding = HuggingFaceEmbeddings()
    vector_db = Chroma.from_documents(documents=docs, embedding=embedding)
    retriever = vector_db.as_retriever()

    result = retriever.invoke("What is YOLO?")
    print("Number of relevant documents: ", len(result))

## Cấu hình mô hình:

    nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
    )

    MODEL_NAME = "lmsys/vicuna-7b-v1.5"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=nf4_config,
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        pad_token_id=tokenizer.eos_token_id,
        device_map="auto"
    )

    llm = HuggingFacePipeline(pipeline=model_pipeline)

## Tạo chuỗi phản hồi:

    def format_docs(docs):
        return "\n\n".join(doc.page_content.strip() for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    USER_QUESTION = "ai là tác giả của file pdf này?"

    output = rag_chain.invoke(USER_QUESTION)

## In toàn bộ kết quả để kiểm tra lỗi:
    print(f"Full output:\n{output}")

    try:
        answer = output.split("Answer :")[1].strip()
        print(answer)
    except IndexError:
        print("Warning: Answer prefix 'Answer: ' not found in output. Check prompt or model generation.")

## Chạy mã nguồn:

Lưu toàn bộ mã nguồn vào một file chatbot.py và chạy bằng lệnh sau:

    python chatbot.py


## Lưu ý: 
Điều chỉnh FILE_PATH và các tham số khác nếu cần thiết để phù hợp với môi trường làm việc của bạn.
