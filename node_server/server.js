const express = require('express');
const bodyParser = require('body-parser');
const axios = require('axios');
const path = require('path');
const morgan = require('morgan');

const app = express();
const port = 3000;

app.use(morgan('dev'));
app.use(bodyParser.urlencoded({ extended: true }));
app.use(bodyParser.json());

// Serve static files from the frontend directory
app.use(express.static(path.join(__dirname, '../frontend')));

// Serve the HTML file
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, '../frontend', 'index.html'));
});

// Serve the prediction page
app.get('/predict', (req, res) => {
  res.sendFile(path.join(__dirname, '../frontend', 'predict.html'));
});

// Serve the admin page
app.get('/admin', (req, res) => {
  res.sendFile(path.join(__dirname, '../frontend', 'admin.html'));
});

app.post('/predict', async (req, res) => {
  try {
    console.log('Received request:', req.body);
    const response = await axios.post('http://localhost:5000/predict', req.body);
    console.log('Prediction response:', response.data);
    res.send(response.data);
  } catch (error) {
    console.error('Error during prediction:', error);
    res.status(500).send(error.toString());
  }
});

app.get('/admin/data', async (req, res) => {
  try {
    const response = await axios.get('http://localhost:5000/admin/data');
    res.send(response.data);
  } catch (error) {
    console.error('Error fetching admin data:', error);
    res.status(500).send(error.toString());
  }
});

app.delete('/admin/data/:id', async (req, res) => {
  try {
    const response = await axios.delete(`http://localhost:5000/admin/data/${req.params.id}`);
    res.send(response.data);
  } catch (error) {
    console.error('Error deleting user data:', error);
    res.status(500).send(error.toString());
  }
});

app.listen(port, () => {
  console.log(`Server is running on http://localhost:${port}`);
});
