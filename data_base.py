import os
from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders.pdf import PyMuPDFLoader
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from zhipuai_embedding import ZhipuAIEmbeddings
from langchain_community.vectorstores import Chroma


def load_env_variables():
    load_dotenv(find_dotenv())


def get_file_paths(folder_path):
    """获取指定文件夹下的所有文件路径。

    Args:
        folder_path (str): 要加载文档的文件夹路径。

    Returns:
        list: 文件路径列表。
    """
    file_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    return file_paths


def create_loaders(file_paths):
    """根据文件类型为每个文件创建相应的加载器。

    Args:
        file_paths (list): 文件路径列表。

    Returns:
        list: 加载器对象列表。
    """
    loaders = []
    for file_path in file_paths:
        file_type = file_path.split('.')[-1]
        if file_type == 'pdf':
            loaders.append(PyMuPDFLoader(file_path))
        elif file_type == 'md':
            loaders.append(UnstructuredMarkdownLoader(file_path))
    return loaders


def load_texts(loaders):
    """加载每个文件的内容。

    Args:
        loaders (list): 加载器对象列表。

    Returns:
        list: 文本内容列表。
    """
    texts = []
    for loader in loaders:
        texts.extend(loader.load())
    if not texts:
        raise ValueError("没有加载到任何文档。请检查文件夹路径和文件类型。")
    return texts


def split_texts(texts, chunk_size=500, chunk_overlap=50):
    """将文本拆分成更小的块。

    Args:
        texts (list): 文本内容列表。
        chunk_size (int, optional): 块大小。默认为500。
        chunk_overlap (int, optional): 块重叠大小。默认为50。

    Returns:
        list: 拆分后的文档列表。
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    split_docs = text_splitter.split_documents(texts)
    if not split_docs:
        raise ValueError("没有拆分任何文档。请检查文本拆分逻辑。")
    return split_docs


def create_and_persist_vectordb(split_docs, embedding, persist_directory):
    """从拆分文档创建向量存储，并持久化到磁盘。

    Args:
        split_docs (list): 拆分后的文档列表。
        embedding (object): 嵌入模型对象。
        persist_directory (str): 向量存储的持久化目录。
    """
    vectordb = Chroma.from_documents(
        documents=split_docs[:20],
        embedding=embedding,
        persist_directory=persist_directory
    )
    print("向量存储已成功创建并持久化。")


def main():
    """主函数，执行文档加载、拆分和向量存储的创建与持久化。"""
    load_env_variables()

    folder_path = 'data_base/knowledge_db'
    file_paths = get_file_paths(folder_path)
    if not file_paths:
        raise ValueError("文件路径列表为空")

    loaders = create_loaders(file_paths)
    texts = load_texts(loaders)

    split_docs = split_texts(texts)

    embedding = ZhipuAIEmbeddings()
    if embedding is None:
        raise ValueError("Embedding 初始化失败")

    persist_directory = 'data_base/vector_db/chroma'

    create_and_persist_vectordb(split_docs, embedding, persist_directory)


if __name__ == "__main__":
    main()
