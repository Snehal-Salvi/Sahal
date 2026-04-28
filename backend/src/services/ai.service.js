import axios from "axios";

function unwrapAIServiceError(error) {
  const responseBody = error.response?.data;

  if (!responseBody) {
    throw error;
  }

  // Axios may have already parsed the JSON body into a plain object
  if (typeof responseBody === "object" && !Buffer.isBuffer(responseBody)) {
    const detail = responseBody?.detail || responseBody?.message;
    throw new Error(detail || JSON.stringify(responseBody));
  }

  try {
    const text = Buffer.from(responseBody).toString("utf8");
    const parsed = JSON.parse(text);
    const detail = parsed?.detail || parsed?.message;
    throw new Error(detail || text);
  } catch (parseError) {
    if (parseError instanceof SyntaxError) {
      throw new Error(Buffer.from(responseBody).toString("utf8"));
    }
    throw parseError;
  }
}

export async function analyzeVideoWithAI({ videoUrl }) {
  try {
    const response = await axios.post(
      `${process.env.AI_SERVICE_URL}/analyze`,
      {
        videoUrl
      },
      {
        timeout: 0
      }
    );

    return response.data;
  } catch (error) {
    unwrapAIServiceError(error);
  }
}

export async function processVideoWithAI({
  videoUrl,
  detectedFaces,
  filterAssignments
}) {
  try {
    const response = await axios.post(
      `${process.env.AI_SERVICE_URL}/process`,
      {
        videoUrl,
        detectedFaces,
        filterAssignments
      },
      {
        responseType: "arraybuffer",
        timeout: 0
      }
    );

    return Buffer.from(response.data);
  } catch (error) {
    unwrapAIServiceError(error);
  }
}
