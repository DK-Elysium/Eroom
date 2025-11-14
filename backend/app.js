const express = require('express');
const dotenv = require('dotenv');
const cors = require('cors');
const bodyParser = require('body-parser');

// 환경 변수 로드
dotenv.config();

// 서버 설정
const app = express();
const port = process.env.PORT || 5000;

app.use(cors());
app.use(bodyParser.json());

// 기본 라우트
app.get('/', (req, res) => {
  res.send('Hello from the backend server!');
});

app.listen(port, () => {
  console.log(`Server running on port ${port}`);
});
