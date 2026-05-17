import { Router } from "express";
import { deleteAccount, googleSignIn } from "../controllers/auth.controller.js";
import { requireAuth } from "../middleware/requireAuth.js";

const router = Router();
const asyncHandler = (handler) => (req, res, next) =>
  Promise.resolve(handler(req, res, next)).catch(next);

router.post("/google", asyncHandler(googleSignIn));
router.delete("/me", requireAuth, asyncHandler(deleteAccount));

export default router;
