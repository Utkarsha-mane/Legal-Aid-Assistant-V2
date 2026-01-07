import requests
from pathlib import Path

BACKEND = 'http://localhost:8000'
PDF_PATH = Path('scripts/test_doc.pdf')

pdf_content = b"%PDF-1.4\n1 0 obj<< /Type /Catalog /Pages 2 0 R>>endobj\n2 0 obj<< /Type /Pages /Kids [3 0 R] /Count 1>>endobj\n3 0 obj<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>endobj\n4 0 obj<< /Length 44 >>stream\nBT /F1 24 Tf 50 150 Td (Hello PDF) Tj ET\nendstream endobj\n5 0 obj<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>endobj\nxref\n0 6\n0000000000 65535 f \n0000000010 00000 n \n0000000061 00000 n \n0000000116 00000 n \n0000000221 00000 n \n0000000300 00000 n \ntrailer<< /Root 1 0 R /Size 6 >>\nstartxref\n360\n%%EOF"

PDF_PATH.write_bytes(pdf_content)
print('Wrote', PDF_PATH)

with open(PDF_PATH, 'rb') as f:
    files = {'file': (PDF_PATH.name, f, 'application/pdf')}
    try:
        r = requests.post(f'{BACKEND}/upload', files=files, timeout=120)
        print('STATUS', r.status_code)
        try:
            print('JSON:', r.json())
        except Exception:
            print('TEXT:', r.text)
    except Exception as e:
        print('ERROR', e)
