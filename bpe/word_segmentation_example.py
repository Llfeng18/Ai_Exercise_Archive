import tiktoken

f_log = open(r'articles.lst', 'r', encoding='utf-8')
log = f_log.read()
f_log.close()
print(len(log))
enc = tiktoken.encoding_for_model("gpt-4")
print(len(enc.encode(log)))
enc = tiktoken.encoding_for_model("gpt-3.5")
print(len(enc.encode(log)))
enc = tiktoken.encoding_for_model("gpt-2")
print(len(enc.encode(log)))