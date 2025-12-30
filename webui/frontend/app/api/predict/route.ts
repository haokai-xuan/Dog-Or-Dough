import { NextRequest, NextResponse } from 'next/server'

export async function POST(req: NextRequest) {
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
