"""
Alert dispatch tools for sending notifications via webhook, email, and Telegram.

What it does:
    Provides three concrete tool implementations for dispatching alerts:
    SendWebhookTool for HTTP POST webhooks, SendEmailTool for SMTP email
    delivery, and SendTelegramTool for Telegram Bot API messages.

Entities in it:
    - SendWebhookTool: Sends alert payloads to webhook URLs via HTTP POST.
    - SendEmailTool: Sends alert emails via SMTP.
    - SendTelegramTool: Sends alert messages via Telegram Bot API.

How used by other modules:
    - Registered in the ToolRegistry at application startup.
    - Called by agents during workflow execution to dispatch alerts.
    - Credentials (webhook URLs, SMTP config, bot tokens) are injected before execution.
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Any

import httpx

from backend.tools.base import BaseTool, ToolExecutionError


class SendWebhookTool(BaseTool):
    """
    Sends alert payloads to webhook URLs via HTTP POST.

    Description:
        Posts JSON payloads to configured webhook endpoints with optional
        custom headers for integration with services like Slack, Discord,
        or custom alert receivers.

    Attributes:
        None specific beyond BaseTool.

    Methods:
        name: Returns 'send_webhook'.
        description: Returns the tool's purpose description.
        parameters_schema: Returns JSON schema for webhook parameters.
        execute: Sends the webhook POST request.
    """

    @property
    def name(self) -> str:
        """
        Unique name for this tool.

        Description:
            Returns the canonical name used for registry lookup.

        Params:
            None

        Returns:
            str: 'send_webhook'
        """
        return "send_webhook"

    @property
    def description(self) -> str:
        """
        Human-readable description of this tool.

        Description:
            Explains the webhook dispatch functionality.

        Params:
            None

        Returns:
            str: Description string.
        """
        return "Sends alert payloads to webhook URLs via HTTP POST with JSON body."

    @property
    def parameters_schema(self) -> dict:
        """
        JSON Schema for the execute() parameters.

        Description:
            Defines url, payload, and headers parameters.

        Params:
            None

        Returns:
            dict: JSON Schema dictionary.
        """
        return {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Webhook endpoint URL"},
                "payload": {"type": "object", "description": "JSON payload to send"},
                "headers": {
                    "type": "object",
                    "description": "Optional custom headers",
                    "additionalProperties": {"type": "string"},
                },
            },
            "required": ["url", "payload"],
        }

    async def execute(self, **kwargs: Any) -> dict:
        """
        Send a webhook POST request with the given payload.

        Description:
            Posts the JSON payload to the specified URL with optional headers.

        Params:
            **kwargs (Any): Must include 'url' and 'payload'. Optional: 'headers'.

        Returns:
            dict: Dictionary with 'status_code' and 'response_body'.

        Raises:
            ToolExecutionError: If the HTTP request fails or returns non-2xx.
        """
        url = kwargs.get("url")
        payload = kwargs.get("payload")
        custom_headers = kwargs.get("headers", {})

        if not url:
            raise ToolExecutionError("Webhook URL is required")
        if payload is None:
            raise ToolExecutionError("Webhook payload is required")

        headers = {"Content-Type": "application/json"}
        headers.update(custom_headers)

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=payload, headers=headers)
                if response.status_code >= 400:
                    raise ToolExecutionError(
                        f"Webhook POST to {url} failed with HTTP {response.status_code}: "
                        f"{response.text}"
                    )
                return {
                    "status_code": response.status_code,
                    "response_body": response.text,
                }
        except httpx.HTTPError as http_error:
            raise ToolExecutionError(
                f"Webhook request to {url} failed: {http_error}"
            ) from http_error


class SendEmailTool(BaseTool):
    """
    Sends alert emails via SMTP.

    Description:
        Composes and sends email messages using SMTP credentials from the
        injected credentials dictionary. Supports HTML and plain text bodies.

    Attributes:
        None specific beyond BaseTool.

    Methods:
        name: Returns 'send_email'.
        description: Returns the tool's purpose description.
        parameters_schema: Returns JSON schema for email parameters.
        execute: Sends the email via SMTP.
    """

    @property
    def name(self) -> str:
        """
        Unique name for this tool.

        Description:
            Returns the canonical name used for registry lookup.

        Params:
            None

        Returns:
            str: 'send_email'
        """
        return "send_email"

    @property
    def description(self) -> str:
        """
        Human-readable description of this tool.

        Description:
            Explains the email dispatch functionality.

        Params:
            None

        Returns:
            str: Description string.
        """
        return "Sends alert emails via SMTP with configurable recipients and content."

    @property
    def parameters_schema(self) -> dict:
        """
        JSON Schema for the execute() parameters.

        Description:
            Defines recipient, subject, body, and format parameters.

        Params:
            None

        Returns:
            dict: JSON Schema dictionary.
        """
        return {
            "type": "object",
            "properties": {
                "to_address": {"type": "string", "description": "Recipient email address"},
                "subject": {"type": "string", "description": "Email subject line"},
                "body": {"type": "string", "description": "Email body content"},
                "body_format": {
                    "type": "string",
                    "enum": ["plain", "html"],
                    "description": "Body content format",
                    "default": "plain",
                },
            },
            "required": ["to_address", "subject", "body"],
        }

    async def execute(self, **kwargs: Any) -> dict:
        """
        Send an email via SMTP using injected credentials.

        Description:
            Connects to the SMTP server specified in credentials and sends
            the composed email message.

        Params:
            **kwargs (Any): Must include 'to_address', 'subject', 'body'.

        Returns:
            dict: Dictionary with 'sent' status and 'recipient'.

        Raises:
            ToolExecutionError: If SMTP credentials are missing or sending fails.
        """
        to_address = kwargs.get("to_address")
        subject = kwargs.get("subject")
        body = kwargs.get("body")
        body_format = kwargs.get("body_format", "plain")

        if not to_address:
            raise ToolExecutionError("Recipient email address (to_address) is required")
        if not subject:
            raise ToolExecutionError("Email subject is required")
        if not body:
            raise ToolExecutionError("Email body is required")

        smtp_host = self.credentials.get("smtp_host")
        smtp_port = self.credentials.get("smtp_port")
        smtp_username = self.credentials.get("smtp_username")
        smtp_password = self.credentials.get("smtp_password")
        from_address = self.credentials.get("from_address")

        if not all([smtp_host, smtp_port, smtp_username, smtp_password, from_address]):
            raise ToolExecutionError(
                "SMTP credentials incomplete. Required: smtp_host, smtp_port, "
                "smtp_username, smtp_password, from_address"
            )

        message = MIMEMultipart()
        message["From"] = from_address
        message["To"] = to_address
        message["Subject"] = subject
        message.attach(MIMEText(body, body_format))

        try:
            with smtplib.SMTP(smtp_host, int(smtp_port)) as server:
                server.starttls()
                server.login(smtp_username, smtp_password)
                server.send_message(message)
        except smtplib.SMTPException as smtp_error:
            raise ToolExecutionError(
                f"Failed to send email to {to_address} via {smtp_host}:{smtp_port}: {smtp_error}"
            ) from smtp_error

        return {"sent": True, "recipient": to_address}


class SendTelegramTool(BaseTool):
    """
    Sends alert messages via Telegram Bot API.

    Description:
        Posts messages to Telegram chats using the Bot API with a bot token
        and chat ID from injected credentials.

    Attributes:
        TELEGRAM_API_BASE: Class-level constant for the Telegram Bot API base URL.

    Methods:
        name: Returns 'send_telegram'.
        description: Returns the tool's purpose description.
        parameters_schema: Returns JSON schema for Telegram parameters.
        execute: Sends the Telegram message.
    """

    TELEGRAM_API_BASE = "https://api.telegram.org"

    @property
    def name(self) -> str:
        """
        Unique name for this tool.

        Description:
            Returns the canonical name used for registry lookup.

        Params:
            None

        Returns:
            str: 'send_telegram'
        """
        return "send_telegram"

    @property
    def description(self) -> str:
        """
        Human-readable description of this tool.

        Description:
            Explains the Telegram message dispatch functionality.

        Params:
            None

        Returns:
            str: Description string.
        """
        return "Sends alert messages to Telegram chats via the Bot API."

    @property
    def parameters_schema(self) -> dict:
        """
        JSON Schema for the execute() parameters.

        Description:
            Defines chat_id, message, and parse_mode parameters.

        Params:
            None

        Returns:
            dict: JSON Schema dictionary.
        """
        return {
            "type": "object",
            "properties": {
                "chat_id": {"type": "string", "description": "Telegram chat ID"},
                "message": {"type": "string", "description": "Message text to send"},
                "parse_mode": {
                    "type": "string",
                    "enum": ["Markdown", "HTML", "MarkdownV2"],
                    "description": "Message formatting mode",
                    "default": "Markdown",
                },
            },
            "required": ["chat_id", "message"],
        }

    async def execute(self, **kwargs: Any) -> dict:
        """
        Send a message to a Telegram chat via the Bot API.

        Description:
            Uses the bot_token from credentials to POST to the sendMessage
            endpoint of the Telegram Bot API.

        Params:
            **kwargs (Any): Must include 'chat_id' and 'message'.

        Returns:
            dict: Dictionary with 'sent' status and 'message_id'.

        Raises:
            ToolExecutionError: If bot_token is missing or the API request fails.
        """
        chat_id = kwargs.get("chat_id")
        message = kwargs.get("message")
        parse_mode = kwargs.get("parse_mode", "Markdown")

        if not chat_id:
            raise ToolExecutionError("Telegram chat_id is required")
        if not message:
            raise ToolExecutionError("Telegram message text is required")

        bot_token = self.credentials.get("bot_token")
        if not bot_token:
            raise ToolExecutionError(
                "Telegram bot_token not found in credentials. "
                "Credentials must be injected with 'bot_token' before sending."
            )

        url = f"{self.TELEGRAM_API_BASE}/bot{bot_token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": parse_mode,
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=payload)
                if response.status_code != 200:
                    raise ToolExecutionError(
                        f"Telegram API error (HTTP {response.status_code}): {response.text}"
                    )
                response_data = response.json()
                if not response_data.get("ok"):
                    raise ToolExecutionError(
                        f"Telegram API returned error: {response_data.get('description', 'Unknown error')}"
                    )
                return {
                    "sent": True,
                    "message_id": response_data["result"]["message_id"],
                }
        except httpx.HTTPError as http_error:
            raise ToolExecutionError(
                f"Telegram API request failed: {http_error}"
            ) from http_error
