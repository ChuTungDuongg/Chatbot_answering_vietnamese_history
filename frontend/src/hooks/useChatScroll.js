import { useEffect, useLayoutEffect, useRef } from "react";

const NEAR_BOTTOM_PX = 120;

export function useChatScroll(messages, status) {
  const scrollerRef = useRef(null);
  const contentRef = useRef(null);
  const followingRef = useRef(true);
  const hasMessages = messages.length > 0;

  const followLatest = () => { followingRef.current = true; };
  const onScroll = (event) => {
    const node = event.currentTarget;
    followingRef.current = node.scrollHeight - node.scrollTop - node.clientHeight <= NEAR_BOTTOM_PX;
  };

  useLayoutEffect(() => {
    const node = scrollerRef.current;
    if (hasMessages && followingRef.current && node) node.scrollTo({ top: node.scrollHeight, behavior: "instant" });
  }, [messages, status, hasMessages]);

  useEffect(() => {
    const node = scrollerRef.current;
    const content = contentRef.current;
    if (!hasMessages || !node || !content) return;
    // Composer growth, fonts and viewport resize can change available height
    // without a new message. Keep following only if the reader already was.
    const observer = new ResizeObserver(() => {
      if (followingRef.current) node.scrollTo({ top: node.scrollHeight, behavior: "instant" });
    });
    observer.observe(node);
    observer.observe(content);
    return () => observer.disconnect();
  }, [hasMessages]);

  return { scrollerRef, contentRef, onScroll, followLatest };
}
