import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import App from "./App";
import { ToastHost } from "./components/ui";
import { applyTheme, useUi } from "./store";
import "./index.css";

// Paint the stored theme before first render so there is no flash.
applyTheme(useUi.getState().theme);

const queryClient = new QueryClient({
  defaultOptions: { queries: { refetchOnWindowFocus: false } },
});

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <ToastHost>
        <App />
      </ToastHost>
    </QueryClientProvider>
  </StrictMode>,
);
