import uuid
from rag_engine import FinanceRAG, Document
doc = Document(id=str(uuid.uuid4()), name="test", text="The company saw a massive decrease in profits this quarter, losing over $100 million. However, revenue was steady.")
rag = FinanceRAG([doc])
print(rag.answer("Did they lose money?"))
