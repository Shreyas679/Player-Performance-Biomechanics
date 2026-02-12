import axios from "axios";
export const api = axios.create({ baseURL: "http://localhost:8000" });
export async function uploadKeypoints(file) {
  const form = new FormData();
  form.append("file", file);
  const res = await api.post("/predict", form, { headers: { "Content-Type": "multipart/form-data" } });
  return res.data;
}
