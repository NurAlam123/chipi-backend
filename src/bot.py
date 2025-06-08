import os
import gc
from queue import Queue
import threading
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer  # type: ignore[reportPrivateImportUsage]


from config import LLM_MODEL_PATH, LLM_MAX_TOKEN
from db import ConversationDB


class StreamingToQueue(TextStreamer):
    def __init__(self, tokenizer, queue, **kwargs):
        super().__init__(
            tokenizer, skip_prompt=True, skip_special_tokens=True, **kwargs
        )
        self.queue = queue

    def on_finalized_text(self, text, stream_end=False):
        self.queue.put(text)
        if stream_end:
            self.queue.put(None)


class Bot:
    def __init__(self, db_config=None) -> None:
        self.model = None
        self.tokenizer = None
        self.history = []
        self.user_input = ""
        self.messages = []
        self.content = None
        self.thinking_content = None
        # Initialize database if config provided
        self.db = ConversationDB(db_config) if db_config else None

    def load_model(self):
        print("Starting model...")
        # ===== Qwen3-0.6B =====
        self.tokenizer = AutoTokenizer.from_pretrained(
            LLM_MODEL_PATH, local_files_only=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_PATH,
            torch_dtype="auto",
            device_map="auto",
            local_files_only=True,
        )
        print("Model started...")
        self.model_loaded = True

    def unload_model(self):
        print("Unloading model...")

        if self.model:
            self.model.to("cpu")
            del self.model
        if self.tokenizer:
            del self.tokenizer

        self.model = None
        self.tokenizer = None
        self.content = None
        self.thinking_content = None

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        print("Model unloaded.")

    def prompt(self, prompt: str, role: str = "user") -> None:
        self.user_input = prompt
        self.messages = self.history + [{"role": role, "content": prompt}]

    def generate(self, stream: bool = False, thinking: bool = True) -> None:
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model is not loaded. Call `load_model()` first.")

        if not self.messages:
            raise AttributeError("Set a prompt first using `prompt()`.")

        text = self.tokenizer.apply_chat_template(
            self.messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking,
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        if stream:
            streamer = TextStreamer(
                self.tokenizer, skip_prompt=True, skip_special_tokens=True
            )
        else:
            streamer = None

        model_response = self.model.generate(
            **inputs,
            max_new_tokens=LLM_MAX_TOKEN,
            streamer=streamer,
        )

        response_ids = model_response[0][len(inputs.input_ids[0]) :].tolist()

        self.history.append({"role": "user", "content": self.user_input})

        # Store in database
        if self.db:
            self.db.add_message(self.session_id, "user", self.user_input, None)

        try:
            index = len(response_ids) - response_ids[::-1].index(151668)
        except ValueError:
            index = 0

        # response = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        thinking_content = self.tokenizer.decode(
            response_ids[:index], skip_special_tokens=True
        ).strip("\n")

        content = self.tokenizer.decode(
            response_ids[index:], skip_special_tokens=True
        ).strip("\n")

        self.history.append({"role": "assistant", "content": content})
        self.thinking_content = thinking_content
        self.content = content

        # Store in database
        if self.db:
            self.db.add_message(self.session_id, "assistant", content, thinking_content)

    def generate_stream(self, q: Queue, thinking: bool = False):
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model is not loaded. Call `load_model()` first.")

        if not self.messages:
            raise AttributeError("Set a prompt first using `prompt()`.")

        streamer = StreamingToQueue(self.tokenizer, q)

        tokenizer = self.tokenizer
        model = self.model

        def run_generate():
            text = tokenizer.apply_chat_template(
                self.messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=thinking,
            )

            inputs = tokenizer(text, return_tensors="pt").to(model.device)

            model_response = model.generate(
                **inputs,
                max_new_tokens=LLM_MAX_TOKEN,
                streamer=streamer,
            )

            response_ids = model_response[0][len(inputs.input_ids[0]) :].tolist()

            self.history.append({"role": "user", "content": self.user_input})

            # Store in database
            if self.db:
                self.db.add_message(self.session_id, "user", self.user_input, None)

            try:
                index = len(response_ids) - response_ids[::-1].index(151668)
            except ValueError:
                index = 0

            thinking_content = tokenizer.decode(
                response_ids[:index], skip_special_tokens=True
            ).strip("\n")

            content = tokenizer.decode(
                response_ids[index:], skip_special_tokens=True
            ).strip("\n")

            self.history.append({"role": "assistant", "content": content})

            # Store in database
            if self.db:
                self.db.add_message(
                    self.session_id, "assistant", content, thinking_content
                )

        t = threading.Thread(target=run_generate)
        t.start()

        while True:
            token = q.get()
            if token is None:
                break
            text = token.replace("\n", "\\n")
            yield f"data: {text}\n\n"

    def get_thinking(self) -> str:
        if self.thinking_content is None:
            raise AttributeError("Call `generate()` first to access thinking content.")

        return self.thinking_content

    def get_content(self) -> str:
        if self.content is None:
            raise AttributeError("Call `generate()` first to access response content.")
        return self.content

    def start_new_conversation(self) -> str:
        """Create a new conversation session and return its ID"""
        if not self.db:
            return ""
        self.session_id = self.db.create_conversation()
        self.history = []
        return self.session_id

    def get_conversation_history(self, session_id: str):
        """Get conversation history from database"""
        if not self.db:
            return
        messages = self.db.get_conversation(session_id)
        return messages or []

    def delete_conversation(self, session_id: str) -> bool:
        """Delete a conversation from database"""
        if not self.db:
            return False
        return self.db.delete_conversation(session_id)


if __name__ == "__main__":

    from multiprocessing import Queue, Process

    def worker_process(prompt):
        try:
            # Load model if not loaded
            if not bot.model or not bot.tokenizer:
                bot.load_model()

            bot.prompt(prompt)
            bot.generate()
            # bot.generate(stream=True, thinking=False)
            res = bot.get_content()
            print(res)

        except Exception:
            raise

    bot = Bot()
    while True:
        # print(conversation)

        prompt = input("Prompt: ")
        if prompt.lower() == "clear":
            os.system("clear")
            continue

        if prompt.lower() in ["quit", "exit", "q"]:
            break

        p = Process(target=worker_process, args=(prompt,), daemon=True)
        p.start()
        # Start the worker process
        p.join()
        if p.is_alive():
            p.terminate()
