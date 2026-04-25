import "dotenv/config";
import "./config/cloudinary.js";
import app from "./app.js";
import { connectDatabase } from "./config/db.js";

const port = Number(process.env.PORT || 5000);

await connectDatabase();

app.listen(port, () => {
  console.log(`API server listening on port ${port}`);
});

