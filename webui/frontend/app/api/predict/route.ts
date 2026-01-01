import { NextRequest, NextResponse } from 'next/server'

const requests = new Map<string, {count: number, resetAt: number}>()
const MAX_REQUESTS = 100
const WINDOW_MS = 60000

export async function POST(req: NextRequest) {
  // Check rate limit
  const ip = req.headers.get("x-forwarded-for")?.split(",")[0] || req.headers.get("x-real-ip") || "unknown"
  const now = Date.now()
  const record = requests.get(ip)

  if (record && now < record.resetAt) {
    if (record.count > MAX_REQUESTS) {
      const retryAfter = Math.ceil((record.resetAt - now) / 1000)
      return NextResponse.json(
        {error: "Rate limit exceeded. Please try again later.", retryAfter},
        {status: 429, headers: {"Retry-After": retryAfter.toString()}}
      )
    }
    record.count++
  }
  else {
    requests.set(ip, {count: 1, resetAt: now + WINDOW_MS})
  }

  const apiUrl = process.env.API_URL
  const apiSecret = process.env.API_SECRET

  if (!apiUrl) {
    return NextResponse.json(
      { error: "API_URL environment variable is not set" },
      { status: 500 }
    )
  }

  if (!apiSecret) {
    return NextResponse.json(
      { error: "API_SECRET environment variable is not set" },
      { status: 500 }
    )
  }

  const formData = await req.formData()

  try {
    // Send request to fastapi backend
    const res = await fetch(`${apiUrl}/predict`, {
      method: "POST",
      headers: {
        "api-key": apiSecret, // secret stays server-side
      },
      body: formData,
    })

    if (!res.ok) {
      return NextResponse.json(
        { error: `Backend request failed: ${res.status} ${res.statusText}` },
        { status: res.status }
      )
    }

    const data = await res.json()
    return NextResponse.json(data)
  } catch (error) {
    return NextResponse.json(
      { error: `Failed to connect to backend: ${error instanceof Error ? error.message : 'Unknown error'}` },
      { status: 500 }
    )
  }
}
